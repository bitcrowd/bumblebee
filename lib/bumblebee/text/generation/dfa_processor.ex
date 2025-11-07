defmodule Bumblebee.Text.Generation.DFAProcessor do
  @moduledoc false

  import Nx.Defn

  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.LogitsProcessor

  options = [
    initial_state: [
      default: nil,
      doc: "the initial state"
    ],
    state_transitions: [
      default: nil,
      doc: "the definition of a deterministic finite automaton used for constrained generation"
    ],
    vocab_size: [
      default: nil,
      doc: "the size of the vocabulary"
    ]
  ]

  defstruct Bumblebee.Shared.option_defaults(options)

  @impl Bumblebee.Configurable
  def config(logits_processor, opts) do
    Bumblebee.Shared.put_config_attrs(logits_processor, opts)
  end

  @impl Bumblebee.LogitsProcessor
  def init(logits_processor, context) do
    dfa = logits_processor

    num_states =
      dfa.state_transitions
      |> Enum.flat_map(fn {state, _token_id, next_state} -> [state, next_state] end)
      |> Enum.uniq()
      |> Enum.count()

    # we add 1 to num_states as we want to have an empty row for state 0
    # 0 should represent "no transition" as this is the only false value in nx
    empty_state_transitions_tensor = Nx.broadcast(0, {num_states + 1, dfa.vocab_size})

    state_transitions_tensor =
      for transition <- dfa.state_transitions, reduce: empty_state_transitions_tensor do
        transitions_tensor ->
          {current_state, token_id, next_state} = transition
          index = Nx.tensor([current_state, token_id])

          Nx.indexed_put(transitions_tensor, index, next_state)
      end

    initial_state = Nx.tensor(dfa.initial_state)
    [initial_state, _sequence] = Nx.broadcast_vectors([initial_state, context.sequence])

    %{
      last_state: initial_state,
      state_transitions_tensor: state_transitions_tensor
    }
  end

  @impl Bumblebee.LogitsProcessor
  def process(_logits_processor, state, logits, context) do
    dfa_processing(logits, state, context)
  end

  deftransform dfa_processing(logits, state, context) do
    transitions_tensor = state.state_transitions_tensor
    last_state = state.last_state

    current_state = current_state(context, last_state, transitions_tensor)
    logits = logits(logits, transitions_tensor, current_state)

    state = %{state | last_state: current_state}

    {state, logits}
  end

  defnp current_state(context, last_state, transitions_tensor) do
    if context.length == context.input_length do
      last_state
    else
      last_token_id = context.sequence[context.length - 1]
      transitions_tensor[[last_state, last_token_id]]
    end
  end

  defnp logits(logits, transitions_tensor, current_state) do
    suppressed_logits = Nx.fill(logits, Nx.Constants.neg_infinity(), type: Nx.type(logits))
    allowed_token_ids = transitions_tensor[current_state]

    Nx.select(allowed_token_ids, logits, suppressed_logits)
  end
end
