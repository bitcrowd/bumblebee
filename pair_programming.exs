Mix.install([
  {:bumblebee, path: "../bumblebee_bitcrowd"},
  {:nx, "~> 0.10.0", override: true},
  {:emlx, github: "elixir-nx/emlx"},
  {:benchee, "~> 1.0"}
])

Nx.global_default_backend({EMLX.Backend, device: :gpu})
repo = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
{:ok, model_info} = Bumblebee.load_model(repo, backend: {EMLX.Backend, device: :gpu})
{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)

sequence_length = 512

prompt = """
Give me an array that contains a mix of numbers and text.
There MUST be at least one number and one text.
Valid examples are:

["hello",89,"hola",6,4,8]
"""

numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
array_start_token = "["
array_end_token = "]"
array_addition_token = ","
# String Token would require ! (like "everything, just without ....)
# Token 18
string_token = "\""

# ToDo: should be a list -> idx
states = [
  :starting,
  :in_array,
  :in_number,
  :in_addition,
  :in_string,
  :end_of_string,
  :ending,
  :done
]

state_to_num = fn state -> Enum.find_index(states, & &1 == state) end

# ------------------------------------- above chars ------------------------------ #
# ------------------------------------- below tokens ------------------------------ #

array_start_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_start_token)
array_end_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_end_token)
addition_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, array_addition_token)
string_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, string_token)
end_of_sequence_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :eos)

special_tokens_ids = for token_id <- 0..17, do: token_id
number_tokens_ids = Enum.map(numbers, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1))
vocabulary_token_ids = for token_id <- 0..model_info.spec.vocab_size, do: token_id

string_token_ids = vocabulary_token_ids -- ([string_token_id] ++ special_tokens_ids)

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
## done (7)

## which tokens lead to which state from given state
state_transitions =
  [
    # starting
    {:starting, [array_start_token_id], :in_array},
    # in_array
    {:in_array, number_tokens_ids, :in_number},
    {:in_array, [array_end_token_id], :ending},
    {:in_array, [string_token_id], :in_string},
    # in_number
    {:in_number, number_tokens_ids, :in_number},
    {:in_number, [addition_token_id], :in_addition},
    {:in_number, [array_end_token_id], :ending},
    # in_addition
    {:in_addition, number_tokens_ids, :in_number},
    {:in_addition, [string_token_id], :in_string},
    # in_string
    {:in_string, string_token_ids, :in_string},
    {:in_string, [string_token_id], :end_of_string},
    # end_of_string
    {:end_of_string, [addition_token_id], :in_addition},
    {:end_of_string, [array_end_token_id], :ending},
    # ending
    {:ending, [end_of_sequence_token_id], :done}
  ]
  |> Enum.flat_map(fn {current_state, tensor_ids, next_state} ->
    for tensor_id <- tensor_ids do
      {state_to_num.(current_state), tensor_id, state_to_num.(next_state)}
    end
  end)

dfa = %{ state_transitions: state_transitions, }

generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 24,
    strategy: %{type: :multinomial_sampling, top_p: 0.6},
    dfa: dfa
  )

serving =
  Bumblebee.Text.generation(model_info, tokenizer, generation_config,
    compile: [batch_size: 1, sequence_length: sequence_length],
    stream: false,
    defn_options: [compiler: Nx.Defn.Evaluator]
  )

%{results: [_result]} =  Nx.Serving.run(serving, prompt) |> dbg

