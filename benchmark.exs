Mix.install([
  {:bumblebee, path: "../bumblebee_bitcrowd"},
  {:nx, "~> 0.10.0", override: true},
  {:exla, "~> 0.10.0"},
  {:emlx, github: "elixir-nx/emlx"},
  {:benchee, "~> 1.0"}
])

# backend = EMLX.Backend
# compiler = Nx.Defn.Evaluator
backend = EXLA.Backend
compiler = EXLA

Nx.global_default_backend(backend)

repo = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}

sequence_length = 512

prompt = """
Give me an array that contains a mix of numbers and text.
There MUST be at least one number and one text.
Valid examples are:

["hello",89,"hola",6,4,8]
"""

# this DFA definition is "array of integers" generatd by outlines-core
# 
# let schema = r#"{
#     "type": "array",
#     "items": {
#         "type": "integer"
#     }
# }"#;

initial_state = 64

state_transitions =
  [
    {96, 33, 128},
    {96, 40, 128},
    {96, 36, 128},
    {96, 32, 112},
    {96, 39, 128},
    {96, 35, 128},
    {96, 38, 128},
    {96, 34, 128},
    {96, 41, 128},
    {96, 37, 128},
    {144, 2, 144},
    {176, 77, 144},
    {224, 33, 240},
    {224, 40, 240},
    {224, 36, 240},
    {224, 32, 112},
    {224, 39, 240},
    {224, 35, 240},
    {224, 38, 240},
    {224, 34, 240},
    {224, 41, 240},
    {224, 37, 240},
    {128, 33, 128},
    {128, 77, 144},
    {128, 36, 128},
    {128, 28, 192},
    {128, 39, 128},
    {128, 10790, 224},
    {128, 34, 128},
    {128, 37, 128},
    {128, 40, 128},
    {128, 32, 128},
    {128, 216, 176},
    {128, 35, 128},
    {128, 38, 128},
    {128, 41, 128},
    {128, 6329, 144},
    {80, 33, 128},
    {80, 77, 144},
    {80, 29, 96},
    {80, 36, 128},
    {80, 41, 128},
    {80, 32, 112},
    {80, 39, 128},
    {80, 216, 176},
    {80, 35, 128},
    {80, 40, 128},
    {80, 38, 128},
    {80, 34, 128},
    {80, 6329, 144},
    {80, 37, 128},
    {112, 216, 176},
    {112, 10790, 224},
    {112, 77, 144},
    {112, 6329, 144},
    {112, 28, 192},
    {64, 9197, 96},
    {64, 75, 160},
    {208, 33, 240},
    {208, 29, 224},
    {208, 36, 240},
    {208, 40, 240},
    {208, 32, 112},
    {208, 39, 240},
    {208, 35, 240},
    {208, 38, 240},
    {208, 34, 240},
    {208, 41, 240},
    {208, 37, 240},
    {160, 33, 128},
    {160, 77, 144},
    {160, 36, 128},
    {160, 39, 128},
    {160, 256, 176},
    {160, 731, 96},
    {160, 34, 128},
    {160, 37, 128},
    {160, 29, 96},
    {160, 40, 128},
    {160, 32, 112},
    {160, 216, 80},
    {160, 35, 128},
    {160, 38, 128},
    {160, 6329, 144},
    {160, 41, 128},
    {240, 33, 240},
    {240, 77, 144},
    {240, 36, 240},
    {240, 28, 192},
    {240, 39, 240},
    {240, 10790, 224},
    {240, 34, 240},
    {240, 37, 240},
    {240, 40, 240},
    {240, 32, 240},
    {240, 216, 176},
    {240, 35, 240},
    {240, 38, 240},
    {240, 41, 240},
    {240, 6329, 144},
    {192, 33, 240},
    {192, 29, 224},
    {192, 36, 240},
    {192, 40, 240},
    {192, 32, 112},
    {192, 39, 240},
    {192, 216, 208},
    {192, 35, 240},
    {192, 731, 224},
    {192, 38, 240},
    {192, 34, 240},
    {192, 41, 240},
    {192, 37, 240}
  ]

unique_states =
  Enum.flat_map(state_transitions, fn {state, _token_id, next_state} -> [state, next_state] end)
  |> Enum.uniq()
  |> Enum.sort()

states_map = for {state, i} <- Enum.with_index(unique_states), into: %{}, do: {state, i}

compact_states =
  Enum.map(state_transitions, fn {state, token_id, next_state} ->
    {states_map[state], token_id, states_map[next_state]}
  end)

state_transitions = compact_states
initial_state = states_map[initial_state]

dfa = %{state_transitions: state_transitions, mode: :stateful, initial_state: initial_state}

build_serving = fn backend, compiler, max_new_tokens, dfa ->
  Nx.global_default_backend(backend)

  {:ok, model_info} = Bumblebee.load_model(repo, backend: backend)

  {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
  {:ok, generation_config} = Bumblebee.load_generation_config(repo)

  generation_config =
    Bumblebee.configure(generation_config,
      max_new_tokens: max_new_tokens,
      min_length: sequence_length + max_new_tokens,
      strategy: %{type: :multinomial_sampling, top_p: 0.6},
      dfa: dfa
    )

    Bumblebee.Text.generation(model_info, tokenizer, generation_config,
      compile: [batch_size: 1, sequence_length: sequence_length],
      stream: false,
      defn_options: [compiler: compiler]
    )
end

Benchee.run(
  %{
    ## regular sampling
    "Regular Sampling, EMLX" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EMLX.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens 
        dfa = nil

        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Regular Sampling, EXLA with Evaluator" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens
        dfa = nil
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Regular Sampling, EXLA with Compiler" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = EXLA
        max_new_tokens = max_new_tokens
        dfa = nil
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    ## stateless constrained sampling
    "Stateless Constrained Sampling, EMLX" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EMLX.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateless}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Stateless Constrained Sampling, EXLA with Evaluator" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateless}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Stateless Constrained Sampling, EXLA with Compiler" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = EXLA
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateless}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    ## stateful constrained sampling
    "Stateful Constrained Sampling, EMLX" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EMLX.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateful}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Stateful Constrained Sampling, EXLA with Evaluator" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = Nx.Defn.Evaluator
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateful}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    },
    "Stateful Constrained Sampling, EXLA with Compiler" => {
      fn {_max_new_tokens, serving} -> Nx.Serving.run(serving, prompt) end,
      before_scenario: fn max_new_tokens ->
        backend = EXLA.Backend
        compiler = EXLA
        max_new_tokens = max_new_tokens
        dfa = %{dfa | mode: :stateful}
        serving = build_serving.(backend, compiler, max_new_tokens, dfa)

        Nx.Serving.run(serving, prompt)

        {max_new_tokens, serving}
      end
    }
  },
  # save: [path: "save.benchee", tag: "first-try"],
  # formatters: [{Benchee.Formatters.Console, comparison: true, extended_statistics: false}],
  time: 60,
  inputs: %{
    "max_new_tokens: 8" => 8,
    "max_new_tokens: 64" => 64
  }
)
