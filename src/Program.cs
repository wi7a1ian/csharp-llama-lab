using LLama.Common;
using LLama;

var model = new LLamaModel(new ModelParams(
    modelPath: Path.Combine("C:", "Models", "wizardLM-7B.ggmlv3.q4_1.bin"),
    contextSize: 1024, 
    seed: 1337, 
    gpuLayerCount: 5));

var ex = new InteractiveExecutor(model);
var session = new ChatSession(ex);

var prompt = "Transcript of a dialogdotnet run, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\r\n\r\nUser: Hello, Bob.\r\nBob: Hello. How may I help you today?\r\nUser: Please tell me the largest city in Europe.\r\nBob: Sure. The largest city in Europe is Moscow, the capital of Russia.\r\nUser:";
Console.WriteLine();
Console.Write(prompt);
while (true)
{
    foreach (var text in session.Chat(prompt, new InferenceParams() { Temperature = 0.6f, AntiPrompts = new List<string> { "User:" } }))
    {
        Console.Write(text);
    }

    Console.ForegroundColor = ConsoleColor.Green;
    prompt = Console.ReadLine();
    Console.ForegroundColor = ConsoleColor.White;
}