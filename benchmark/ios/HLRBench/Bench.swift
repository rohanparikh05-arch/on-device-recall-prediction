// Times both Core ML models and checks their predictions against reference
// values. The same file runs on the iPhone and, through mac/main.swift, on the
// Mac, so the two timings come from one harness.

import CoreML
import Foundation

let batchSizes = [1, 32, 256, 1024]
let warmupCalls = 20
let trialCalls = 500
let rounds = 3

struct ModelRows: Decodable {
    let idx: [Int32]
    let p: [Double]
    let h: [Double]
    let table_size: Int
}

struct Rows: Decodable {
    let right: [Float]
    let wrong: [Float]
    let t: [Float]
    let neural: ModelRows
    let linear: ModelRows
}

func floatArray(_ values: [Float]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .float32)
    for (i, v) in values.enumerated() { array[i] = NSNumber(value: v) }
    return array
}

func intArray(_ values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .int32)
    for (i, v) in values.enumerated() { array[i] = NSNumber(value: v) }
    return array
}

func features(right: [Float], wrong: [Float], t: [Float], idx: [Int32]) throws -> MLFeatureProvider {
    try MLDictionaryFeatureProvider(dictionary: [
        "right": MLFeatureValue(multiArray: try floatArray(right)),
        "wrong": MLFeatureValue(multiArray: try floatArray(wrong)),
        "t": MLFeatureValue(multiArray: try floatArray(t)),
        "lexeme_idx": MLFeatureValue(multiArray: try intArray(idx)),
    ])
}

func randomInput(batch: Int, tableSize: Int) throws -> MLFeatureProvider {
    // same ranges as coreml_neural_benchmark_latency.py
    try features(
        right: (0..<batch).map { _ in Float(Int.random(in: 0..<20)) },
        wrong: (0..<batch).map { _ in Float(Int.random(in: 0..<20)) },
        t: (0..<batch).map { _ in Float.random(in: 0.01..<30.0) },
        idx: (0..<batch).map { _ in Int32.random(in: 0..<Int32(tableSize)) })
}

func timeCalls(model: MLModel, input: MLFeatureProvider, trials: Int) throws -> [Double] {
    for _ in 0..<warmupCalls { _ = try model.prediction(from: input) }
    var latenciesMs: [Double] = []
    for _ in 0..<trials {
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try model.prediction(from: input)
        // read a value so the work can't be deferred past the timer
        _ = output.featureValue(for: "p")!.multiArrayValue![0]
        latenciesMs.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
    }
    return latenciesMs
}

func maxError(model: MLModel, rows: Rows, reference: ModelRows) throws -> (dp: Double, dh: Double) {
    let output = try model.prediction(from: try features(
        right: rows.right, wrong: rows.wrong, t: rows.t, idx: reference.idx))
    let p = output.featureValue(for: "p")!.multiArrayValue!
    let h = output.featureValue(for: "h")!.multiArrayValue!
    var dp = 0.0, dh = 0.0
    for i in 0..<rows.t.count {
        dp = max(dp, abs(p[i].doubleValue - reference.p[i]))
        dh = max(dh, abs(h[i].doubleValue - reference.h[i]))
    }
    return (dp, dh)
}

func memoryMB() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return (Double(info.phys_footprint) / 1e6, Double(info.ledger_phys_footprint_peak) / 1e6)
}

func deviceInfo() -> [String: Any] {
    var system = utsname()
    uname(&system)
    let machine = withUnsafeBytes(of: &system.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
    return [
        "machine": machine,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
    ]
}

func thermalName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func loadModel(_ url: URL, units: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = units
    return try MLModel(contentsOf: url, configuration: config)
}

// memory runs load one model into a fresh process, so the models can't share a figure
func runMemory(name: String, modelURL: URL, tableSize: Int) throws -> [String: Any] {
    let before = memoryMB()
    let model = try loadModel(modelURL, units: .all)
    let loaded = memoryMB()
    let input = try randomInput(batch: 1, tableSize: tableSize)
    for _ in 0..<100 { _ = try model.prediction(from: input) }
    let after = memoryMB()
    return ["device": deviceInfo(), "model": name, "before_mb": before.now, "after_load_mb": loaded.now,
            "after_predictions_mb": after.now, "peak_mb": after.peak]
}

func runBenchmark(neuralURL: URL, linearURL: URL, rowsURL: URL) throws -> [String: Any] {
    let rows = try JSONDecoder().decode(Rows.self, from: Data(contentsOf: rowsURL))
    var results: [String: Any] = ["device": deviceInfo(), "thermal_start": thermalName()]
    let models: [(String, URL, ModelRows)] = [("neural", neuralURL, rows.neural), ("linear", linearURL, rows.linear)]

    for (unitName, units) in [("all", MLComputeUnits.all), ("cpu_only", MLComputeUnits.cpuOnly)] {
        var perUnit: [String: Any] = [:]
        var loaded: [String: MLModel] = [:]
        for (name, url, reference) in models {
            let model = try loadModel(url, units: units)
            loaded[name] = model
            let error = try maxError(model: model, rows: rows, reference: reference)
            perUnit["\(name)_fidelity"] = ["max_dp": error.dp, "max_dh": error.dh]
        }
        var latencies: [String: [Double]] = [:]
        for _ in 0..<rounds {
            for (name, _, reference) in models {
                for batch in batchSizes {
                    let trials = batch <= 32 ? trialCalls : trialCalls / 4
                    let input = try randomInput(batch: batch, tableSize: reference.table_size)
                    latencies["\(name)_\(batch)", default: []] += try timeCalls(model: loaded[name]!, input: input, trials: trials)
                }
            }
        }
        for (key, values) in latencies {
            let sorted = values.sorted()
            perUnit[key] = ["mean_ms": values.reduce(0, +) / Double(values.count),
                            "median_ms": sorted[sorted.count / 2],
                            "p99_ms": sorted[min(sorted.count - 1, sorted.count * 99 / 100)],
                            "n": values.count]
        }
        results[unitName] = perUnit
    }
    results["thermal_end"] = thermalName()
    return results
}

func report(_ results: [String: Any], to url: URL) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url)
    return String(decoding: data, as: UTF8.self)
}

// Which rows disagree with the reference, and does it depend on batch size?
func runDiagnostic(neuralURL: URL, linearURL: URL, rowsURL: URL) throws -> [String: Any] {
    let rows = try JSONDecoder().decode(Rows.self, from: Data(contentsOf: rowsURL))
    var results: [String: Any] = ["device": deviceInfo()]
    for (name, url, reference) in [("neural", neuralURL, rows.neural), ("linear", linearURL, rows.linear)] {
        let model = try loadModel(url, units: .cpuOnly)
        let output = try model.prediction(from: try features(right: rows.right, wrong: rows.wrong, t: rows.t, idx: reference.idx))
        let p = output.featureValue(for: "p")!.multiArrayValue!
        let h = output.featureValue(for: "h")!.multiArrayValue!
        var bad: [Int] = []
        var worstDh = 0.0
        for i in 0..<rows.t.count {
            let dh = abs(h[i].doubleValue - reference.h[i])
            if dh > 1e-3 || abs(p[i].doubleValue - reference.p[i]) > 1e-3 { bad.append(i) }
            worstDh = max(worstDh, dh)
        }
        var examples: [[String: Any]] = []
        for i in bad.prefix(6) {
            let single = try model.prediction(from: try features(right: [rows.right[i]], wrong: [rows.wrong[i]], t: [rows.t[i]], idx: [reference.idx[i]]))
            examples.append(["row": i, "right": rows.right[i], "wrong": rows.wrong[i], "t": rows.t[i], "idx": reference.idx[i],
                             "ref_p": reference.p[i], "ref_h": reference.h[i],
                             "batch_p": p[i].doubleValue, "batch_h": h[i].doubleValue,
                             "single_p": single.featureValue(for: "p")!.multiArrayValue![0].doubleValue,
                             "single_h": single.featureValue(for: "h")!.multiArrayValue![0].doubleValue])
        }
        results[name] = ["bad_rows": bad.count, "first_bad": Array(bad.prefix(40)), "worst_dh": worstDh,
                         "shape": p.shape.map { $0.intValue }, "strides": p.strides.map { $0.intValue },
                         "h_shape": h.shape.map { $0.intValue }, "h_strides": h.strides.map { $0.intValue },
                         "datatype": p.dataType.rawValue, "examples": examples]
    }
    return results
}
