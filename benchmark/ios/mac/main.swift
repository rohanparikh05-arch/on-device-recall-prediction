// Runs Bench.swift on the Mac, to compare the phone with the laptop through
// the same harness. Usage: mac_bench <compiled model dir> <rows.json> <output.json> [mem-neural|mem-linear]

import Foundation

let args = CommandLine.arguments
let modelDir = URL(fileURLWithPath: args[1])
let neural = modelDir.appendingPathComponent("NeuralHLR.mlmodelc")
let linear = modelDir.appendingPathComponent("LinearHLR.mlmodelc")
let output = URL(fileURLWithPath: args[3])

if args.contains("mem-neural") {
    print(try report(try runMemory(name: "neural", modelURL: neural, tableSize: 19172), to: output))
} else if args.contains("mem-linear") {
    print(try report(try runMemory(name: "linear", modelURL: linear, tableSize: 19172), to: output))
} else {
    print(try report(try runBenchmark(neuralURL: neural, linearURL: linear, rowsURL: URL(fileURLWithPath: args[2])), to: output))
}
