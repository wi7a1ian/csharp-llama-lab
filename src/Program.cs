using LLama.Common;
using LLama;

var modelPath = Path.Combine("C:", "Models", "wizardLM-7B.ggmlv3.q4_1.bin");
//LLamaQuantizer.Quantize(modelPath, Path.ChangeExtension(modelPath, ".q4_1.bin"), "q4_1");

var model = new LLamaModel(new ModelParams(
    modelPath: modelPath,
    contextSize: 1024, 
    seed: 1337, 
    gpuLayerCount: 5));

int[] carTokens = model.Tokenize("car").ToArray();
int[] redCarTokens = model.Tokenize("red car").ToArray();

var embedder = new LLamaEmbedder(new ModelParams(modelPath));
float[] carEembeddings = embedder.GetEmbeddings("car");
float[] redCarEmbeddings = embedder.GetEmbeddings("red car");

// InteractiveExecutor - "chat-with-bob", stateful, completing the prompt, 
// InstructExecutor - "alpaca", stateful, completing the prompt, 
// StatelessExecutor - no state, always provide whole prompt for context, better use "Q:... A:..." otherwise it won't stop until char limit is reached

var executor = new InteractiveExecutor(model); // "chat-with-bob"
var session = new ChatSession(executor); // maintains the context of the conversation

if(new []{"model.st", "executor.st", "session.st"}.All(File.Exists))
{
    session.LoadSession("session.st"); // achieves the same as below:
    //model.LoadState("model.st");
    //executor.LoadState("executor.st");
}

// Save chat history when done with interactive session
Console.CancelKeyPress += (s, e) => {
    using var sw = new StreamWriter("history.log");
    foreach(var rec in session.History.Messages)
    {
        sw.WriteLine($"{rec.AuthorRole}: {rec.Content}");
    }
};

// Add input prompt transformations (e.g for sanitization)
// session.AddInputTransform(new MyInputTransform1()).AddInputTransform(new MyInputTransform2());
// session.WithOutputTransform(new MyOutputTransform());

var prompt = "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is a lawyer, helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\r\n\r\nUser: Hello, Bob.\r\nBob: Hello. How may I help you today?\r\nUser: Please tell me what GDRP is.\r\nBob:";
Console.WriteLine();
Console.Write(prompt);
while (true)
{
    foreach (var text in session.Chat(prompt, new InferenceParams() { 
        Temperature = 0.6f, 
        AntiPrompts = new List<string> { "User:" },
        // PathSession = "" // path to file for saving/loading model eval state
    }))
    {
        Console.Write(text);
    }

    // var stateData = model.GetStateData();
    // model.LoadState(stateData);
    
    //session.SaveSession("session.st"); // achieves the same as below:
    //model.SaveState("model.st");
    //executor.SaveState("executor.st");

    Console.ForegroundColor = ConsoleColor.Green;
    prompt = Console.ReadLine(); // User: Is there something similar in United States?
    Console.ForegroundColor = ConsoleColor.White;
}