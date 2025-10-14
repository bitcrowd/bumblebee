Mix.install([
  {:bumblebee, path: "../bumblebee_bitcrowd"},
  {:nx, "~> 0.10.0", override: true},
  {:emlx, github: "elixir-nx/emlx"}
])

Nx.global_default_backend({EMLX.Backend, device: :gpu})
repo = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
{:ok, model_info} = Bumblebee.load_model(repo, backend: {EMLX.Backend, device: :gpu})
{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)

sequence_length = 512

prompt = """
Give me 10 random, single digit numbers in an array.
Valid examples are:

[8,2,3,8,6,4,8,6,4,8]
"""

numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
start_token = ["["]
end_token = ["]"]
addition_token = [","]

# states
# * start_token -> start -> in_number, end -> numbers ++ end_token
# * numbers -> in_number -> in_number, addition, end -> numbers ++ addition_token ++ end_token
# * addition_token -> addition -> in_number -> numbers
# * end_token -> end -> END_OF_SEQUENCE -> END_OF_SEQUENCE

# {
#   start: start_token,
#   numbers: numbers,
# }

## am anfang war nix
## -> Start token
## last_state = inspect_last_token oder last state from stack

## Am Anfang sind wir im start_token state und haben den start token schon
## die nächsten kandidaten wählen
## die große wahl
## inpect current token -> determine which state was chosen
## next loop

## last token -> state

end_of_sequence_token = Bumblebee.Tokenizer.special_token(tokenizer, :eos)

states_to_num = %{
  starting: 0,
  in_array: 1,
  in_number: 2,
  in_addition: 3,
  ending: 4
}

## transitions
transitions = %{
  starting: Enum.map(start_token, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
  in_array: Enum.map(numbers ++ end_token, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
  in_number:
    Enum.map(
      numbers ++ addition_token ++ end_token,
      &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)
    ),
  in_addition: Enum.map(numbers, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
  ending: Enum.map([end_of_sequence_token], &Bumblebee.Tokenizer.token_to_id(tokenizer, &1))
}

## states
states =
  %{
    starting: [],
    in_array: Enum.map(start_token, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
    in_number: Enum.map(numbers, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
    in_addition: Enum.map(addition_token, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)),
    ending: Enum.map(end_token, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1))
  }
  |> Enum.flat_map(fn {state, tensor_ids} ->
    for tensor_id <- tensor_ids do
      {tensor_id, states_to_num[state]}
    end
  end)

dfa = %{
  states: states,
  transitions: transitions
}

generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 48,
    strategy: %{type: :multinomial_sampling, top_p: 0.6},
    dfa: dfa
  )

serving =
  Bumblebee.Text.generation(model_info, tokenizer, generation_config,
    compile: [batch_size: 1, sequence_length: sequence_length],
    stream: false,
    defn_options: [compiler: Nx.Defn.Evaluator]
  )

{:ok, _pid} =
  Supervisor.start_link([{Nx.Serving, name: Serving, serving: serving}], strategy: :one_for_one)

Nx.Serving.run(serving, prompt) |> dbg
