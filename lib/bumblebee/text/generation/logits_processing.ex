defmodule Bumblebee.Text.Generation.LogitsProcessing do
  @moduledoc false

  import Nx.Defn

  deftransform dfa_processor(logits, context, opts \\ []) do
    opts = Keyword.validate!(opts, [:dfa])
    dfa = opts[:dfa]

    transitions_max_length = Map.values(dfa.transitions) |> Enum.map(&length(&1)) |> Enum.max()

    transitions_tensor =
      dfa.transitions
      |> Enum.with_index()
      |> Enum.map(fn {{_state, token_ids}, index} -> {index, Nx.tensor(token_ids)} end)
      |> Enum.map(fn {_idx, tensor} ->
        Nx.pad(tensor, -1, [{0, transitions_max_length - Nx.size(tensor), 0}])
      end)
      |> Nx.stack()

    states_tensor = Nx.broadcast(-1, {Nx.size(logits)})

    states_indices = Enum.map(dfa.states, fn {key, _value} -> key end) |> Nx.tensor()
    states_indices = Nx.new_axis(states_indices, -1)

    states_states = Enum.map(dfa.states, fn {_key, state} -> state end) |> Nx.tensor()

    states_tensor = Nx.indexed_put(states_tensor, states_indices, states_states)

    state_logits(logits, context, transitions_tensor, states_tensor)
  end

  defn state_logits(logits, context, transitions_tensor, states_tensor) do
    ## states tensor
    ## 0 -> -1
    ## ...
    ## 75 -> 1 (in_array)
    ## ..
    current_state =
      if context.length == context.input_length do
        0
      else
        last_token = context.sequence[context.length - 1]
        states_tensor[last_token]
      end

    ## figure out allowed tokens for next sampling

    ## transition tensor
    ## 0 -> token_ids for starting (padded to largest dimension)
    ## 1 -> token_ids for in_array (padded to largest dimension)
    ## ...
    allowed_tokens = transitions_tensor[current_state]

    ## pass allowed tokens into allow_token_ids
    # allow_token_ids(logits, allowed_tokens)
    allow_token_ids(logits, allowed_tokens)
  end

  ## figure out state from last token in context
  # deftransform figure_out_state(context, dfa) do
  #   # _last_token = context.sequence[0]

  #   if context.length == 512 do
  #     :starting
  #   else
  #     :in_number
  #     last_token = context.sequence[context.length - 1]

  #     {state, _tokens} =
  #       dfa.states
  #       |> Enum.find(fn {_state, tokens} ->
  #         last_token == nil || last_token in tokens
  #       end)

  #     state
  #   end
  # end

  deftransform suppressed_tokens_processor(logits, _context, opts \\ []) do
    opts = Keyword.validate!(opts, [:suppressed_token_ids])

    indices = opts[:suppressed_token_ids] |> Nx.tensor() |> Nx.new_axis(-1)
    values = Nx.broadcast(Nx.Constants.neg_infinity(Nx.type(logits)), {Nx.size(indices)})
    Nx.indexed_put(logits, indices, values)
  end

  deftransform allowed_tokens_processor(logits, _context, opts \\ []) do
    _opts = Keyword.validate!(opts, [:allowed_token_ids])

    allow_token_ids(logits, opts[:allowed_token_ids])
  end

  defn bos_token_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:bos_token_id])
    bos_token_id = opts[:bos_token_id]

    if context.length == 1 do
      force_token_id(logits, bos_token_id)
    else
      logits
    end
  end

  defn eos_token_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id])
    eos_token_id = opts[:eos_token_id]

    max_length = Nx.axis_size(context.sequence, 0)

    if context.length == max_length - 1 do
      force_token_id(logits, eos_token_id)
    else
      logits
    end
  end

  defn forced_tokens_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:forced_token_ids])
    forced_token_ids(logits, context, opts[:forced_token_ids])
  end

  deftransformp forced_token_ids(logits, context, forced_token_ids) do
    clauses =
      for {idx, token_id} <- forced_token_ids do
        {Nx.equal(context.length, idx), force_token_id(logits, token_id)}
      end

    # Note that we can't use defn ifs inside transform, so we build
    # the expression directly
    Nx.Defn.Expr.cond(clauses, logits)
  end

  defn min_length_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_ids, :min_length_fun])
    eos_token_ids = opts[:eos_token_ids]
    min_length_fun = opts[:min_length_fun]

    min_length = min_length_fun.(context.input_length)

    if context.length < min_length do
      ignore_token_ids(logits, eos_token_ids)
    else
      logits
    end
  end

  defn no_repeat_ngram_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:ngram_length])
    ngram_length = opts[:ngram_length]

    if context.length + 1 < ngram_length do
      logits
    else
      # Given a sequence of last {ngram_length - 1} tokens, we look
      # for prior occurrences of that sequence and we want to make the
      # subsequent token ignored. This way the n-gram is not repeated
      # this time around

      ngram_but_one_length = ngram_length - 1

      last_ngram_but_one =
        Nx.slice_along_axis(
          context.sequence,
          context.length - ngram_but_one_length,
          ngram_but_one_length,
          axis: 0
        )

      {_, _, _, _, logits} =
        while {i = 0, last_ngram_but_one, sequence = context.sequence, length = context.length,
               logits},
              i + ngram_but_one_length < length do
          ngram_but_one = Nx.slice_along_axis(sequence, i, ngram_but_one_length, axis: 0)

          token_id = sequence[i + ngram_but_one_length]
          indices = Nx.new_axis(token_id, -1)

          match? = Nx.all(ngram_but_one == last_ngram_but_one)
          updates = Nx.select(match?, Nx.Constants.neg_infinity(Nx.type(logits)), 0)
          logits = Nx.indexed_add(logits, indices, updates)

          {i + 1, last_ngram_but_one, sequence, length, logits}
        end

      logits
    end
  end

  deftransformp force_token_id(logits, token_id) do
    logits
    |> Nx.fill(Nx.Constants.neg_infinity(), type: Nx.type(logits))
    |> Nx.put_slice([token_id], Nx.tensor([0], type: Nx.type(logits)))
  end

  deftransformp allow_token_ids(logits, allowed_token_ids) do
    # Convert allowed_token_ids to a tensor if it's a list
    allowed_indices = Nx.tensor(allowed_token_ids)
    allowed_logits = Nx.take(logits, allowed_indices)
    suppressed_logits = Nx.fill(logits, Nx.Constants.neg_infinity(), type: Nx.type(logits))

    indices = Nx.new_axis(allowed_indices, -1)
    Nx.indexed_put(suppressed_logits, indices, allowed_logits)
  end

  deftransformp ignore_token_id(logits, token_id) do
    Nx.put_slice(
      logits,
      [token_id],
      Nx.broadcast(Nx.Constants.neg_infinity(Nx.type(logits)), {1})
    )
  end

  deftransformp ignore_token_ids(logits, token_ids) do
    Enum.reduce(token_ids, logits, &ignore_token_id(&2, &1))
  end

  defn temperature_processor(logits, _context, opts \\ []) do
    opts = keyword!(opts, [:temperature])
    temperature = opts[:temperature]

    logits / temperature
  end

  # Processors manipulating the probability distribution

  defn top_k_processor(logits, _context, opts \\ []) do
    opts = keyword!(opts, [:top_k])
    top_k = opts[:top_k]

    {top_k_logits, _} = Nx.top_k(logits, k: top_k)
    kth_logit = top_k_logits[-1]
    Nx.select(logits < kth_logit, Nx.Constants.neg_infinity(Nx.type(logits)), logits)
  end

  defn top_p_processor(logits, _context, opts \\ []) do
    opts = keyword!(opts, [:top_p])
    top_p = opts[:top_p]

    sorted_idx = Nx.argsort(logits)

    cumulative_scores =
      logits
      |> Nx.take_along_axis(sorted_idx)
      |> Axon.Activations.softmax()
      |> Nx.cumulative_sum()

    ordered_ignore_mask = cumulative_scores <= 1 - top_p

    # Arrange the mask back into the original logits order
    ignore_mask =
      Nx.indexed_put(
        Nx.fill(ordered_ignore_mask, 0),
        Nx.new_axis(sorted_idx, -1),
        Nx.flatten(ordered_ignore_mask)
      )

    Nx.select(ignore_mask, Nx.Constants.neg_infinity(Nx.type(logits)), logits)
  end

  defn whisper_timestamp_processor(logits, context, opts \\ []) do
    opts =
      keyword!(opts, [
        :eos_token_id,
        :forced_token_ids,
        :no_timestamps_token_id,
        :timestamp_begin_id
      ])

    eos_token_id = opts[:eos_token_id]
    no_timestamps_token_id = opts[:no_timestamps_token_id]
    timestamp_begin_id = opts[:timestamp_begin_id]

    begin_idx = begin_idx(opts[:forced_token_ids])

    # Ensure the no-timestamps token is never taken
    logits = ignore_token_id(logits, no_timestamps_token_id)

    cond do
      context.length < begin_idx ->
        logits

      context.length == begin_idx ->
        # Output the starting timestamp
        force_token_id(logits, timestamp_begin_id)

      true ->
        logits
        |> force_timestamp_pair(context, begin_idx, eos_token_id, timestamp_begin_id)
        |> maybe_force_timestamp(timestamp_begin_id)
    end
  end

  defnp force_timestamp_pair(logits, context, begin_idx, eos_token_id, timestamp_begin_id) do
    # Force timestamp tokens to appear in pairs, end followed by
    # start, except directly before the EOS token

    prev_was_timestamp? =
      if context.length - begin_idx >= 1 do
        context.sequence[context.length - 1] >= timestamp_begin_id
      else
        Nx.tensor(false)
      end

    # Either second to last was timestamp or is out of range
    prev_was_timestamp_start? =
      if context.length - begin_idx >= 2 do
        context.sequence[context.length - 2] >= timestamp_begin_id
      else
        Nx.tensor(true)
      end

    ignore_mask =
      Nx.logical_and(
        prev_was_timestamp?,
        Nx.select(
          Nx.broadcast(Nx.new_axis(prev_was_timestamp_start?, -1), Nx.shape(logits)),
          # Force non-timestamp
          Nx.iota(Nx.shape(logits)) >= timestamp_begin_id,
          # Force non-normal token
          Nx.iota(Nx.shape(logits)) < eos_token_id
        )
      )

    Nx.select(ignore_mask, Nx.Constants.neg_infinity(Nx.type(logits)), logits)
  end

  defnp maybe_force_timestamp(logits, timestamp_begin_id) do
    # If the total probability of all timestamps exceeds any regular
    # token, we force a timestamp

    log_probabilities = Axon.Activations.log_softmax(logits)
    timestamp_log_probabilities = log_probabilities[timestamp_begin_id..-1//1]
    # TODO: use log_sumexp on Axon v0.6.1
    # timestamp_log_probability = Axon.Activations.log_sumexp(timestamp_log_probabilities)
    timestamp_log_probability = timestamp_log_probabilities |> Nx.exp() |> Nx.sum() |> Nx.log()
    token_log_probabilities = log_probabilities[0..(timestamp_begin_id - 1)//1]

    max_token_log_probability = Nx.reduce_max(token_log_probabilities)
    force_timestamp_mask = timestamp_log_probability > max_token_log_probability
    tokens_mask = Nx.iota(Nx.shape(logits)) < timestamp_begin_id
    ignore_mask = force_timestamp_mask and tokens_mask
    Nx.select(ignore_mask, Nx.Constants.neg_infinity(Nx.type(logits)), logits)
  end

  deftransformp begin_idx(forced_token_ids) do
    case List.last(forced_token_ids) do
      nil -> 1
      {idx, _token_id} -> idx + 1
    end
  end
end
