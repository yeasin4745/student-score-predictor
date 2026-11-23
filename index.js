import fs from "fs";
import path from "path";
import crypto from "crypto";
import readline from "readline/promises";
import { stdin as input, stdout as output } from "process";
import * as tf from "@tensorflow/tfjs";

const CSV_PATH = path.resolve("students_dataset.csv");
const ARTIFACT_PATH = path.resolve("model_artifacts.json");

function getFileHash(filePath) {
  const content = fs.readFileSync(filePath, "utf-8");
  return crypto.createHash("sha256").update(content).digest("hex");
}

function parseCSV(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  const lines = raw.split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const values = lines[i].split(",").map((v) => v.trim());
    if (values.length < headers.length) continue;

    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx];
    });

    const studyHour = parseFloat(row.study_hour);
    const classPresent = parseFloat(row.class_present);
    const sleepHour = parseFloat(row.sleep_hour);
    const previousScore = parseFloat(row.previous_score);
    const targetScore = parseFloat(row.target_score);

    if (
      Number.isFinite(studyHour) &&
      Number.isFinite(classPresent) &&
      Number.isFinite(sleepHour) &&
      Number.isFinite(previousScore) &&
      Number.isFinite(targetScore)
    ) {
      records.push({
        studentId: row.Student_ID || `STU-${i}`,
        features: [studyHour, classPresent, sleepHour, previousScore],
        target: targetScore,
      });
    }
  }

  return records;
}

function computeScaler(featureMatrix) {
  const numCols = featureMatrix[0].length;
  const min = new Array(numCols).fill(Infinity);
  const max = new Array(numCols).fill(-Infinity);

  for (const row of featureMatrix) {
    for (let c = 0; c < numCols; c++) {
      if (row[c] < min[c]) min[c] = row[c];
      if (row[c] > max[c]) max[c] = row[c];
    }
  }

  return { min, max };
}

function normalizeFeatures(featureMatrix, scaler) {
  return featureMatrix.map((row) =>
    row.map((val, idx) => {
      const range = scaler.max[idx] - scaler.min[idx];
      return range === 0 ? 0 : (val - scaler.min[idx]) / range;
    })
  );
}

function createModel(inputDim) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [inputDim],
      units: 16,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 8,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "linear",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  return model;
}

async function saveArtifacts(model, scaler, datasetHash) {
  let savedArtifacts;

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      savedArtifacts = {
        modelTopology: artifacts.modelTopology,
        weightSpecs: artifacts.weightSpecs,
        weightDataBase64: Buffer.from(artifacts.weightData).toString("base64"),
      };
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          totalBytes: artifacts.weightData.byteLength,
        },
      };
    })
  );

  const payload = {
    datasetHash,
    scaler,
    model: savedArtifacts,
  };

  fs.writeFileSync(ARTIFACT_PATH, JSON.stringify(payload), "utf-8");
}

async function loadArtifacts() {
  const raw = fs.readFileSync(ARTIFACT_PATH, "utf-8");
  const payload = JSON.parse(raw);

  const weightBuffer = Buffer.from(payload.model.weightDataBase64, "base64");
  const arrayBuffer = weightBuffer.buffer.slice(
    weightBuffer.byteOffset,
    weightBuffer.byteOffset + weightBuffer.byteLength
  );

  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: payload.model.modelTopology,
      weightSpecs: payload.model.weightSpecs,
      weightData: arrayBuffer,
    })
  );

  return {
    model,
    scaler: payload.scaler,
    datasetHash: payload.datasetHash,
  };
}

async function trainAndSave(datasetHash) {
  const records = parseCSV(CSV_PATH);
  if (records.length === 0) {
    throw new Error("CSV file has no valid training records.");
  }

  const rawFeatures = records.map((r) => r.features);
  const rawTargets = records.map((r) => [r.target / 100]);

  const scaler = computeScaler(rawFeatures);
  const scaledFeatures = normalizeFeatures(rawFeatures, scaler);

  const xs = tf.tensor2d(scaledFeatures);
  const ys = tf.tensor2d(rawTargets);

  const model = createModel(4);

  await model.fit(xs, ys, {
    epochs: 120,
    batchSize: 16,
    shuffle: true,
    verbose: 0,
  });

  xs.dispose();
  ys.dispose();

  await saveArtifacts(model, scaler, datasetHash);
  return { model, scaler };
}

async function getReadyModel() {
  if (!fs.existsSync(CSV_PATH)) {
    throw new Error(`Dataset not found at ${CSV_PATH}`);
  }

  const currentHash = getFileHash(CSV_PATH);

  if (fs.existsSync(ARTIFACT_PATH)) {
    try {
      const loaded = await loadArtifacts();
      if (loaded.datasetHash === currentHash) {
        return { model: loaded.model, scaler: loaded.scaler, retrained: false };
      }
    } catch {
      const trained = await trainAndSave(currentHash);
      return { model: trained.model, scaler: trained.scaler, retrained: true };
    }
  }

  const trained = await trainAndSave(currentHash);
  return { model: trained.model, scaler: trained.scaler, retrained: true };
}

function predictScore(model, scaler, features) {
  const [scaledRow] = normalizeFeatures([features], scaler);
  const inputTensor = tf.tensor2d([scaledRow]);
  const predTensor = model.predict(inputTensor);
  const rawPred = predTensor.dataSync()[0] * 100;
  inputTensor.dispose();
  predTensor.dispose();
  const clamped = Math.min(100, Math.max(0, rawPred));
  return Number(clamped.toFixed(2));
}

async function startCLI(model, scaler) {
  const rl = readline.createInterface({ input, output });

  while (true) {
    const studentId = (await rl.question("\nStudent ID (or 'exit'): ")).trim();
    if (studentId.toLowerCase() === "exit") {
      break;
    }

    const studyHourStr = await rl.question("Study Hour: ");
    const classPresentStr = await rl.question("Class Present (0-100): ");
    const sleepHourStr = await rl.question("Sleep Hour: ");
    const previousScoreStr = await rl.question("Previous Score (0-100): ");

    const studyHour = parseFloat(studyHourStr);
    const classPresent = parseFloat(classPresentStr);
    const sleepHour = parseFloat(sleepHourStr);
    const previousScore = parseFloat(previousScoreStr);

    if (
      !Number.isFinite(studyHour) ||
      !Number.isFinite(classPresent) ||
      !Number.isFinite(sleepHour) ||
      !Number.isFinite(previousScore)
    ) {
      console.log("Invalid numeric input. Please try again.");
      continue;
    }

    const score = predictScore(model, scaler, [
      studyHour,
      classPresent,
      sleepHour,
      previousScore,
    ]);

    console.log(`Student ID: ${studentId || "N/A"} | Predicted Score: ${score}`);
  }

  rl.close();
}

async function main() {
  const { model, scaler, retrained } = await getReadyModel();
  if (retrained) {
    console.log("Model trained and saved.");
  } else {
    console.log("Loaded existing trained model.");
  }
  await startCLI(model, scaler);
}

main();
