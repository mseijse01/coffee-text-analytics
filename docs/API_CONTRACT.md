# Coffee Analytics API — Contract

Base URL: `http://localhost:8000` (local) | `http://<host>:8000` (Docker)

Start the server: `make serve` (requires trained artifacts in `models/`)

---

## Endpoints

### `GET /health`

Returns whether the model is loaded and ready.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `GET /model-info`

Returns metadata about the loaded model.

**Response**
```json
{
  "model_name": "xgboost",
  "r2_score": 0.9453,
  "n_features_selected": 279,
  "feature_extractors": ["tfidf", "bert", "topics", "sentiment"]
}
```

---

### `POST /predict`

Predicts coffee quality rating from review text.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `desc_1` | string | yes | Primary tasting notes |
| `desc_2` | string | no (default `""`) | Secondary review notes |
| `desc_3` | string | no (default `""`) | Bottom-line conclusion |
| `roast` | string | no (default `""`) | Roast level e.g. `"Light-Medium"` |
| `country_of_origin` | string | no (default `""`) | e.g. `"Ethiopia"` |
| `roaster` | string | no (default `""`) | Roaster name |
| `aroma` | float 0–10 | no | Sensory score |
| `acid` | float 0–10 | no | Sensory score |
| `body` | float 0–10 | no | Sensory score |
| `flavor` | float 0–10 | no | Sensory score |
| `aftertaste` | float 0–10 | no | Sensory score |

Only `desc_1` is required. All other fields improve prediction accuracy but are optional — missing categoricals are zero-filled, missing sensory scores are omitted from the feature set.

**Minimal request example**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"desc_1": "bright citrus acidity with floral jasmine notes"}'
```

**Full request example**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "desc_1": "bright citrus acidity with floral jasmine notes",
    "desc_2": "clean finish juicy body",
    "desc_3": "elegant and complex",
    "roast": "Light-Medium",
    "country_of_origin": "Ethiopia",
    "roaster": "Blue Bottle Coffee"
  }'
```

**Response**
```json
{
  "rating": 93.14,
  "model_name": "xgboost",
  "r2_score": 0.9453
}
```

| Field | Type | Description |
|---|---|---|
| `rating` | float | Predicted rating on 80–100 scale |
| `model_name` | string | Model used for prediction |
| `r2_score` | float | Model R² on held-out test set |

---

## Notes for Integration

- **Startup time**: ~30–60 seconds (GloVe loads 400k word vectors). Wait for `"Startup complete. API ready."` in server logs before sending requests.
- **`/health` check**: Use this to confirm the server is ready before sending prediction requests.
- **Rating scale**: 80–100. Scores below 84 are rare; 90+ indicates exceptional quality.
- **Text quality matters**: `desc_1` is the most important field. Longer, more descriptive tasting notes produce more reliable predictions.
- **Missing categoricals**: If `roast`, `country_of_origin`, or `roaster` are unknown, omit them or pass `""` — the model zero-fills and still returns a valid prediction.
