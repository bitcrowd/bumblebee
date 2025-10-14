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
array_start_token = ["["]
array_end_token = ["]"]
array_addition_token = [","]
# String Token would require ! (like "everything, just without ....)
string_token ="\"" # Token 18


# ToDo: should be a list -> idx
states_to_num = %{
  starting: 0,
  in_array: 1,
  in_number: 2,
  in_addition: 3,
  in_string: 4,
  end_of_string: 5,
  ending: 6
}

# ------------------------------------- above chars ------------------------------ #
# ------------------------------------- below tokens ------------------------------ #

array_start_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_start_token))
array_end_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_end_token))
addition_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_addition_token)
string_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, string_token)
end_of_sequence_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :eos)

special_tokens_ids = for token_id <- 0..17, do: token_id
number_tokens_ids = Enum.map(numbers, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1))
vocabulary_token_ids = for token_id <- 0..model_info.vocabulary_size, do: token_id

string_token_ids = vocabulary_token_ids -- [string_token_id] -- special_tokens_ids 

## transitions
transitions = %{
  starting: [array_start_token_id],
  in_array: number_tokens_ids ++ [array_end_token_id, string_token_id], # todo start string token
  in_number: number_tokens_ids ++ [addition_token_id, end_token_id],
  in_addition: number_tokens_ids ++ [string_token_id],
  in_string: string_token_ids ++ [string_token_id],
  end_of_string: [addition_token_id, array_end_token_id],
  ending: [end_of_sequence_token_id]
}

## sequence : 75, 33, 34, ...

# State             0   1  
# chosen Token id  75  18  
# new state         1   3  

## tensor
# State/token ids -> new state
## State / Token ids   0  1  2  ... 18 ... 33 ... 75  76       
## starting (0)       -1 -1 -1      -1     -1      1
## in_array (1)                      4      2          6 
## in_number (2)
## in_addition (3)
## in_string (4)
## end_of_string (5)
## ending (6)


## states
states =
  %{
    starting: [],
    in_array: [array_start_token_id],
    in_number: number_token_ids,
    in_addition: [addition_token_id],
    in_string: [string_token_id],
    end_of_string: [string_token_id],
    ending: [array_end_token_id]
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
