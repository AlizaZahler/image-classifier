# Local image classifier

From the Images Dataset folder:

```bash
source .venv/bin/activate
python serve.py
```

Open http://localhost:8000. Click the upload icon or drag in an image. The page shows the predicted class and all three model scores as percentages.

Uses `runs/20261021-204523/final.pt` by default. To select another model or port:

```bash
python serve.py --checkpoint runs/another-run/final.pt --port 8001
```

The server loads the model once, uses the same preprocessing as prediction, and runs on MPS when available. No new Python dependencies are required. It listens only on your computer's loopback interface. Uploaded images stay in memory and are not saved. PNG, JPEG and WebP are recommended; limit 20 MB / 25 megapixels.

Stop with Control-C in the terminal running the server. Percentages are uncalibrated softmax scores and may differ from 100% by a rounding fraction when added.
