using System;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;

namespace ChatAppGenAI
{
    public class ModelRunner : IDisposable
    {
        private readonly string ModelDir = Path.Combine("C:\\Users\\samkemp\\source\\repos\\ChatAppGenAI\\optimized-model-maths\\model");

        private Model? model = null;
        private Tokenizer? tokenizer = null;
        public event EventHandler? ModelLoaded = null;

        [MemberNotNullWhen(true, nameof(model), nameof(tokenizer))]
        public bool IsReady => model != null && tokenizer != null;

        public void Dispose()
        {
            model?.Dispose();
            tokenizer?.Dispose();
        }

        public IAsyncEnumerable<string> InferStreaming(string systemPrompt, string userPrompt, [EnumeratorCancellation] CancellationToken ct = default)
        {
            var prompt = $@"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{systemPrompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{userPrompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
            return InferStreaming(prompt, ct);
        }
        
        public IAsyncEnumerable<string> InferStreaming(string systemPrompt, List<ModelMessage> history, string userPrompt, [EnumeratorCancellation] CancellationToken ct = default)
        {
            var prompt = $@"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{systemPrompt}<|eot_id|>";
            foreach (var message in history)
            {
                prompt += $"<|start_header_id|>{message.Type.ToString().ToLower()}<|end_header_id|>\n{message.Text}<|eot_id|>";
            }
            prompt += "<|start_header_id|>assistant<|end_header_id|>";

            return InferStreaming(prompt, ct);

        }
        public async IAsyncEnumerable<string> InferStreaming(string prompt, [EnumeratorCancellation] CancellationToken ct = default)
        {
            if (!IsReady)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);
            generatorParams.TryGraphCaptureWithMaxBatchSize(1);

            using var tokenizerStream = tokenizer.CreateStream();
            using var generator = new Generator(model, generatorParams);
            StringBuilder stringBuilder = new();
            while (!generator.IsDone())
            {
                string part;
                try
                {
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }

                    await Task.Delay(0, ct).ConfigureAwait(false);
                    generator.ComputeLogits();
                    generator.GenerateNextToken();
                    part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
                    stringBuilder.Append(part);
             
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex);
                    break;
                }

                yield return part;
            }
        }

        public Task InitializeAsync()
        {
            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                model = new Model(ModelDir);
                tokenizer = new Tokenizer(model);
                sw.Stop();
                Debug.WriteLine($"Model loading took {sw.ElapsedMilliseconds} ms");
                ModelLoaded?.Invoke(this, EventArgs.Empty);
            });
        }
    }

    public class ModelMessage
    {
        public string Text { get; set; }
        public ModelMessageType Type { get; set; }

        public ModelMessage(string text, ModelMessageType type)
        {
            Text = text;
            Type = type;
        }
    }

    public enum  ModelMessageType
    {
        User,
        Assistant
    }
}