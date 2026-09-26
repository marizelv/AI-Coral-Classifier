# MBV Coral AI — GitHub + Vercel

Mobile-friendly coral classification web app using TensorFlow.js.

## 1. Convert the Keras model

In Google Colab:

```python
!pip -q install tensorflowjs
!tensorflowjs_converter --input_format=keras \
    /content/your_model.h5 \
    /content/tfjs_model
```

The result contains:

- `model.json`
- `group1-shard*.bin`

## 2. Copy the model

Copy the contents of `tfjs_model/` into:

```text
public/model/
```

The final structure should be:

```text
public/model/model.json
public/model/group1-shard1ofN.bin
public/model/group1-shard2ofN.bin
...
```

## 3. Important preprocessing

This app currently assumes:

- input size: 256 x 256
- RGB image
- pixel normalization: image / 255.0

If your trained model used different preprocessing, change `app.js` to exactly match training.

## 4. Confidence threshold

The example uses:

```javascript
const CONFIDENCE_THRESHOLD = 0.70;
```

For the dissertation, replace 0.70 with the threshold selected from the validation set.

Do not select the threshold using the test set.

## 5. Local test

Because browsers may block model loading from a `file://` URL, serve the folder through a local web server.

For example:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## 6. GitHub

```bash
git init
git add .
git commit -m "Initial MBV Coral AI mobile classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mbv-coral-ai.git
git push -u origin main
```

## 7. Vercel

Import the GitHub repository into Vercel.

For this static project:

- Framework: Other / static
- Root directory: `.`
- Build command: leave empty
- Output directory: `.`
- Deploy

After deployment, Vercel provides a `.vercel.app` URL.

## 8. Android

Open the Vercel URL in Chrome on Android.

Use:

```text
Chrome → menu → Add to Home screen
```

The site then behaves like a simple installed web app.

## 9. Dissertation deployment description

Suggested wording:

"The optimized CNN model was converted from Keras HDF5 format to TensorFlow.js Layers format and deployed as a mobile-responsive progressive web application. The application performs client-side inference in the browser, accepting images from the device camera or local storage. The system resizes input images to 256 × 256 pixels and applies the same normalization used during model development. A validation-derived confidence threshold is applied to distinguish accepted predictions from unknown or uncertain inputs. The web application was version-controlled using GitHub and deployed through Vercel."

## 10. Future RAG integration

The next module can connect:

```text
CNN prediction
      ↓
species name
      ↓
corpus / FAISS retrieval
      ↓
retrieved scientific information
      ↓
LLM explanation
```

Keep the CNN classifier working first before adding RAG.
