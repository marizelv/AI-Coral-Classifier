const MODEL_URL = "/model/model.json";
const CLASS_URL = "./classes.json";

// Replace this with the threshold selected from your VALIDATION set.
const CONFIDENCE_THRESHOLD = 0.95;

const classes = [
  "Acropora cervicornis",
  "Acropora palmata",
  "Colpophyllia natans",
  "Diadema antillarum",
  "Diploria strigosa",
  "Gorgonians",
  "Millepora alcicornis",
  "Montastraea cavernosa",
  "Meandrina meandrites",
  "Montipora spp.",
  "Palythoas palythoa",
  "Sponges",
  "Siderastrea siderea",
  "Tunicates"
];

let model = null;
let selectedImage = null;

const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");

async function loadModel() {
  try {
    statusEl.textContent = "Loading TensorFlow.js model...";
    await tf.ready();
    model = await tf.loadLayersModel(MODEL_URL);
    statusEl.textContent = "Model loaded. Select a coral image.";
    console.log("Model input shape:", model.inputs[0].shape);
    console.log("Model output shape:", model.outputs[0].shape);
  } catch (error) {
    console.error(error);
    statusEl.textContent =
      "Model could not be loaded. Check public/model/model.json and its .bin files.";
  }
}

input.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  selectedImage = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  analyzeBtn.disabled = !model;
  statusEl.textContent = "Image ready.";
});

analyzeBtn.addEventListener("click", async () => {
  if (!model || !selectedImage) return;

  analyzeBtn.disabled = true;
  statusEl.textContent = "Analyzing...";

  try {
    const image = await createImageBitmap(selectedImage);

    const inputTensor = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(image);
      return pixels
        .resizeBilinear([256, 256])
        .toFloat()
        .div(255.0)
        .expandDims(0);
    });

    const prediction = tf.tidy(() => model.predict(inputTensor));
    const probabilities = await prediction.data();

    inputTensor.dispose();
    prediction.dispose();
    image.close();

    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
    }

    const confidence = probabilities[bestIndex];
    const species = classes[bestIndex] ?? `Class ${bestIndex}`;

    showResult(species, confidence);
  } catch (error) {
    console.error(error);
    statusEl.textContent = "Prediction failed. Check the browser console.";
  } finally {
    analyzeBtn.disabled = false;
  }
});

function showResult(species, confidence) {
  const resultCard = document.getElementById("resultCard");
  const speciesEl = document.getElementById("species");
  const confidenceEl = document.getElementById("confidence");
  const barFill = document.getElementById("barFill");
  const decisionEl = document.getElementById("decision");
  const descriptionEl = document.getElementById("description");

  const pct = confidence * 100;
  speciesEl.textContent = confidence >= CONFIDENCE_THRESHOLD
    ? species
    : "Unknown / Uncertain";

  confidenceEl.textContent = `${pct.toFixed(2)}% confidence`;
  barFill.style.width = `${Math.min(pct, 100)}%`;

  if (confidence >= CONFIDENCE_THRESHOLD) {
    decisionEl.textContent = `Accepted prediction (threshold: ${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%).`;
    descriptionEl.textContent = getDescription(species);
  } else {
    decisionEl.textContent =
      `Below the ${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}% threshold. Treat as unknown/uncertain.`;
    descriptionEl.textContent =
      "The model does not have sufficient confidence to assign this image to one of the known classes.";
  }

  resultCard.hidden = false;
  statusEl.textContent = "Analysis complete.";
}

function getDescription(species) {
  const descriptions = {
    "Acropora cervicornis":
      "Branching staghorn coral category used by the classifier.",
    "Acropora palmata":
      "Elkhorn coral category used by the classifier.",
    "Colpophyllia natans":
      "Boulder-brain coral category used by the classifier.",
    "Diadema antillarum":
      "Long-spined sea urchin category used by the classifier.",
    "Diploria strigosa":
      "Grooved-brain coral category used by the classifier.",
    "Gorgonians":
      "Gorgonian soft-coral category used by the classifier.",
    "Millepora alcicornis":
      "Fire-coral category used by the classifier.",
    "Montastraea cavernosa":
      "Large-cup coral category used by the classifier.",
    "Meandrina meandrites":
      "Maze coral category used by the classifier.",
    "Montipora spp.":
      "Montipora coral category used by the classifier.",
    "Palythoas palythoa":
      "Palythoa category used by the classifier.",
    "Sponges":
      "Sponge category used by the classifier.",
    "Siderastrea siderea":
      "Starlet coral category used by the classifier.",
    "Tunicates":
      "Tunicate category used by the classifier."
  };
  return descriptions[species] || "No description available yet.";
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(console.error);
  });
}

loadModel();
