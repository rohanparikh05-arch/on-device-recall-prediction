import SwiftUI

@main
struct HLRBenchApp: App {
    @State private var status = "Running..."

    var body: some Scene {
        WindowGroup {
            Text(status).font(.system(.footnote, design: .monospaced)).padding()
                .onAppear {
                    UIApplication.shared.isIdleTimerDisabled = true
                    DispatchQueue.global(qos: .userInitiated).async {
                        let text = run()
                        print(text)
                        fflush(stdout)
                        DispatchQueue.main.async { status = String(text.prefix(1500)) }
                    }
                }
        }
    }

    func run() -> String {
        let bundle = Bundle.main
        let neural = bundle.url(forResource: "NeuralHLR", withExtension: "mlmodelc")!
        let linear = bundle.url(forResource: "LinearHLR", withExtension: "mlmodelc")!
        let rows = bundle.url(forResource: "fidelity_rows", withExtension: "json")!
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let arguments = CommandLine.arguments
        do {
            if arguments.contains("diag") {
                return try report(try runDiagnostic(neuralURL: neural, linearURL: linear, rowsURL: rows),
                                  to: documents.appendingPathComponent("diag.json"))
            }
            if arguments.contains("mem-neural") {
                return try report(try runMemory(name: "neural", modelURL: neural, tableSize: 19172),
                                  to: documents.appendingPathComponent("memory_neural.json"))
            }
            if arguments.contains("mem-linear") {
                return try report(try runMemory(name: "linear", modelURL: linear, tableSize: 19172),
                                  to: documents.appendingPathComponent("memory_linear.json"))
            }
            return try report(try runBenchmark(neuralURL: neural, linearURL: linear, rowsURL: rows),
                              to: documents.appendingPathComponent("results.json"))
        } catch {
            return "FAILED: \(error)"
        }
    }
}
