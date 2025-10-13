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

[1]
[4,7]
[2,4,1]
[4,5,3,6]
[1,7,8,0]
[9,4,7,3,5,2]
[8,2,3,8,6,4,8]
[3,5,9]
[8,9,6,7]
"""

allowed_tokens = ["[", "1", "2", ",", "]"]

special_token_ids =
  Bumblebee.Tokenizer.all_special_tokens(tokenizer)
  |> Enum.map(&Bumblebee.Tokenizer.token_to_id(tokenizer, &1))
  |> Enum.reject(&is_nil/1)

allowed_token_ids = Enum.map(allowed_tokens, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1))

all_allowed_token_ids = special_token_ids ++ allowed_token_ids

generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 24,
    allowed_token_ids: all_allowed_token_ids,
    strategy: %{type: :multinomial_sampling, top_p: 0.6}
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
