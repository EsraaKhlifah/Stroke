# api.py - Production Ready Flask API with Face Landmarks Visualization (SAME AS YOUR ORIGINAL)
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
import base64
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def create_visualization(img_rgb, landmarks, h, w, metrics, thresholds, result):
    """Create image with face landmarks visualization - SAME AS YOUR ORIGINAL CODE"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Show image
    axes[0].imshow(img_rgb)

    # Draw mouth (red) - SAME AS ORIGINAL
    mouth_left = (int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h))
    mouth_right = (int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h))
    axes[0].scatter([mouth_left[0], mouth_right[0]], [mouth_left[1], mouth_right[1]],
                    c='red', s=250, marker='o', label='Mouth corners', zorder=5)
    axes[0].plot([mouth_left[0], mouth_right[0]], [mouth_left[1], mouth_right[1]],
                 'r-', linewidth=3, alpha=0.8)

    avg_mouth_y = (mouth_left[1] + mouth_right[1]) / 2
    axes[0].axhline(y=avg_mouth_y, color='red', linestyle='--', alpha=0.5, label='Mouth reference')

    if metrics['mouth_angle_deg'] > thresholds['angle_deg']:
        axes[0].text(mouth_left[0], mouth_left[1] - 20, f'Tilt: {metrics["mouth_angle_deg"]:.1f}°',
                     color='red', fontsize=10, fontweight='bold')

    # Draw eye openness (blue) - SAME AS ORIGINAL
    left_eye_top = (int(landmarks.landmark[159].x * w), int(landmarks.landmark[159].y * h))
    left_eye_bottom = (int(landmarks.landmark[145].x * w), int(landmarks.landmark[145].y * h))
    axes[0].plot([left_eye_top[0], left_eye_bottom[0]], [left_eye_top[1], left_eye_bottom[1]],
                 'b-', linewidth=2, alpha=0.7)
    axes[0].scatter(left_eye_top[0], left_eye_top[1], c='blue', s=100, marker='s')
    axes[0].scatter(left_eye_bottom[0], left_eye_bottom[1], c='blue', s=100, marker='s')

    right_eye_top = (int(landmarks.landmark[386].x * w), int(landmarks.landmark[386].y * h))
    right_eye_bottom = (int(landmarks.landmark[374].x * w), int(landmarks.landmark[374].y * h))
    axes[0].plot([right_eye_top[0], right_eye_bottom[0]], [right_eye_top[1], right_eye_bottom[1]],
                 'b-', linewidth=2, alpha=0.7)
    axes[0].scatter(right_eye_top[0], right_eye_top[1], c='blue', s=100, marker='s')
    axes[0].scatter(right_eye_bottom[0], right_eye_bottom[1], c='blue', s=100, marker='s')

    # Draw eye corners (cyan) - SAME AS ORIGINAL
    inner_left = (int(landmarks.landmark[133].x * w), int(landmarks.landmark[133].y * h))
    inner_right = (int(landmarks.landmark[362].x * w), int(landmarks.landmark[362].y * h))
    outer_left = (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h))
    outer_right = (int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h))

    axes[0].scatter([inner_left[0], inner_right[0]], [inner_left[1], inner_right[1]],
                    c='cyan', s=120, marker='^', label='Eye inner corners')
    axes[0].scatter([outer_left[0], outer_right[0]], [outer_left[1], outer_right[1]],
                    c='cyan', s=120, marker='v', label='Eye outer corners')

    # Draw eyebrows (green) - SAME AS ORIGINAL
    eyebrow_left = (int(landmarks.landmark[63].x * w), int(landmarks.landmark[63].y * h))
    eyebrow_right = (int(landmarks.landmark[293].x * w), int(landmarks.landmark[293].y * h))
    axes[0].scatter([eyebrow_left[0], eyebrow_right[0]], [eyebrow_left[1], eyebrow_right[1]],
                    c='green', s=150, marker='^', label='Eyebrow centers')

    # Add eye openness text
    left_eye_openness = abs(landmarks.landmark[159].y - landmarks.landmark[145].y) * h
    right_eye_openness = abs(landmarks.landmark[386].y - landmarks.landmark[374].y) * h
    axes[0].text(left_eye_top[0], left_eye_top[1] - 10, f'L:{left_eye_openness:.0f}px', fontsize=8, color='blue')
    axes[0].text(right_eye_top[0], right_eye_top[1] - 10, f'R:{right_eye_openness:.0f}px', fontsize=8, color='blue')

    axes[0].set_title(
        f"FACE ANALYSIS - HIGH SENSITIVITY MODE\nMouth threshold: {thresholds['mouth_px']}px | Eye threshold: {thresholds['eye_px']}px",
        fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].axis('off')

    # Bar chart - SAME AS ORIGINAL
    metric_names = ['Mouth\nVertical', 'Mouth\nAngle', 'Eye\nOpenness', 'Inner Eye', 'Outer Eye', 'Eyebrow']
    metric_values = [
        metrics['mouth_vertical_px'],
        metrics['mouth_angle_deg'],
        metrics['eye_asymmetry_px'],
        metrics['inner_eye_px'],
        metrics['outer_eye_px'],
        metrics['eyebrow_px']
    ]
    thresholds_display = [
        thresholds['mouth_px'],
        thresholds['angle_deg'],
        thresholds['eye_px'],
        4, 4, 4
    ]
    weights = [4, 2, 2, 1, 1, 0.5]

    colors = ['red' if v > t else 'green' for v, t in zip(metric_values, thresholds_display)]

    axes[1].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[1].axhline(y=thresholds['mouth_px'], color='darkred', linestyle='--',
                    label=f'Mouth threshold ({thresholds["mouth_px"]}px)', linewidth=1.5)
    axes[1].axhline(y=thresholds['eye_px'], color='orange', linestyle='--',
                    label=f'Eye threshold ({thresholds["eye_px"]}px)', linewidth=1.5)
    axes[1].set_ylabel('Difference (pixels / degrees)')
    axes[1].set_title(
        f'ASYMMETRY ANALYSIS (Stroke Score: {result["stroke_score"]})\n⚠️ HIGH SENSITIVITY - Low mouth threshold')
    axes[1].legend(fontsize=8)

    for i, (val, w) in enumerate(zip(metric_values, weights)):
        axes[1].text(i, val + (max(metric_values) * 0.02), f'{val:.1f}\n(w={w})', ha='center', fontsize=8)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def analyze_facial_symmetry_from_image(img):
    """YOUR ORIGINAL CODE - WITH VISUALIZATION"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0]

    def get_point(idx):
        return (int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h))

    def get_y_pixel(idx):
        return landmarks.landmark[idx].y * h

    # Mouth asymmetry
    mouth_left = get_point(61)
    mouth_right = get_point(291)
    mouth_vertical_diff = abs(mouth_left[1] - mouth_right[1])
    mouth_angle = np.degrees(np.arctan2(mouth_right[1] - mouth_left[1], mouth_right[0] - mouth_left[0]))
    mouth_angle_deviation = abs(mouth_angle)

    # Eye openness
    left_eye_top = get_y_pixel(159)
    left_eye_bottom = get_y_pixel(145)
    left_eye_openness = abs(left_eye_top - left_eye_bottom)

    right_eye_top = get_y_pixel(386)
    right_eye_bottom = get_y_pixel(374)
    right_eye_openness = abs(right_eye_top - right_eye_bottom)
    eye_openness_diff = abs(left_eye_openness - right_eye_openness)

    # Eye corners
    inner_eye_left = get_y_pixel(133)
    inner_eye_right = get_y_pixel(362)
    inner_eye_diff = abs(inner_eye_left - inner_eye_right)

    outer_eye_left = get_y_pixel(33)
    outer_eye_right = get_y_pixel(263)
    outer_eye_diff = abs(outer_eye_left - outer_eye_right)

    # Eyebrows
    eyebrow_left = get_y_pixel(63)
    eyebrow_right = get_y_pixel(293)
    eyebrow_diff = abs(eyebrow_left - eyebrow_right)

    # Thresholds
    face_height = h
    if face_height < 300:
        mouth_threshold = 2
        eye_threshold = 2
        general_threshold = 3
    elif face_height < 600:
        mouth_threshold = 3
        eye_threshold = 3
        general_threshold = 4
    else:
        mouth_threshold = 4
        eye_threshold = 4
        general_threshold = 5

    angle_threshold = 3.0

    # Calculate score
    issues = []
    stroke_score = 0
    mouth_abnormal = False

    if mouth_vertical_diff > mouth_threshold:
        issues.append(
            f"🔴 CRITICAL: Mouth vertical deviation: {mouth_vertical_diff:.1f}px (threshold: {mouth_threshold})")
        stroke_score += 4
        mouth_abnormal = True

    if mouth_angle_deviation > angle_threshold:
        issues.append(f"🔴 Mouth angle tilt: {mouth_angle_deviation:.1f}° from horizontal")
        stroke_score += 2
        mouth_abnormal = True

    if mouth_abnormal:
        stroke_score += 1
        issues.append(f"⚠️ MOUTH ASYMMETRY DETECTED - High stroke indicator")

    if eye_openness_diff > eye_threshold:
        issues.append(f"⚠️ Eye openness asymmetry: {eye_openness_diff:.1f}px")
        stroke_score += 2

    if inner_eye_diff > general_threshold:
        issues.append(f"⚠️ Inner eye corner asymmetry: {inner_eye_diff:.1f}px > {general_threshold}")
        stroke_score += 1

    if outer_eye_diff > general_threshold:
        issues.append(f"⚠️ Outer eye corner asymmetry: {outer_eye_diff:.1f}px > {general_threshold}")
        stroke_score += 1

    if eyebrow_diff > general_threshold:
        issues.append(f"⚠️ Eyebrow asymmetry: {eyebrow_diff:.1f}px > {general_threshold}")
        stroke_score += 0.5

    # Decision
    if mouth_abnormal:
        result_class = "STROKE"
        reason = f"MOUTH ASYMMETRY detected - Vertical: {mouth_vertical_diff:.1f}px, Angle: {mouth_angle_deviation:.1f}°"
        confidence = min(85 + stroke_score * 5, 98)
    elif stroke_score >= 2:
        result_class = "STROKE"
        reason = f"Multiple asymmetry signs (score: {stroke_score}) - Medical consultation recommended"
        confidence = min(65 + stroke_score * 8, 90)
    else:
        result_class = "NO STROKE"
        reason = f"All asymmetries within normal range (score: {stroke_score})"
        confidence = max(85 - stroke_score * 10, 70)

    metrics = {
        "mouth_vertical_px": round(mouth_vertical_diff, 1),
        "mouth_angle_deg": round(mouth_angle_deviation, 1),
        "eye_asymmetry_px": round(eye_openness_diff, 1),
        "inner_eye_px": round(inner_eye_diff, 1),
        "outer_eye_px": round(outer_eye_diff, 1),
        "eyebrow_px": round(eyebrow_diff, 1)
    }

    thresholds_dict = {
        "mouth_px": mouth_threshold,
        "eye_px": eye_threshold,
        "angle_deg": angle_threshold
    }

    result_dict = {
        "result": result_class,
        "confidence": round(confidence, 1),
        "stroke_score": round(stroke_score, 1),
        "reason": reason,
        "metrics": metrics,
        "thresholds": thresholds_dict,
        "face_height_px": face_height,
        "issues": issues
    }

    # Create visualization
    try:
        visualization = create_visualization(img_rgb, landmarks, h, w, metrics, thresholds_dict, result_dict)
    except Exception as e:
        print(f"Visualization error: {e}")
        visualization = None

    return {
        **result_dict,
        "visualization": visualization
    }


# HTML Template with Visualization
TEST_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stroke Detection API - Test Page</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed white;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            background: rgba(255,255,255,0.1);
        }
        input[type="file"] { display: none; }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover { background: #45a049; }
        .preview {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 15px;
        }
        .result {
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .stroke { border-left: 5px solid #ff4444; }
        .no-stroke { border-left: 5px solid #44ff44; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .metric {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: white;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .visualization {
            text-align: center;
            margin-top: 20px;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .endpoints {
            font-size: 12px;
            color: rgba(255,255,255,0.7);
            text-align: center;
            margin-top: 20px;
        }
        .badge {
            background: #007bff;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 10px;
            display: inline-block;
            margin: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🩺 Stroke Detection API</h1>
            <p style="color: white; text-align: center;">Test the API before Flutter integration</p>

            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📸 Click to select image</p>
                <p style="font-size: 12px;">or drag & drop</p>
            </div>
            <input type="file" id="fileInput" accept="image/*">
            <div style="text-align: center;">
                <img id="preview" class="preview" style="display: none;">
            </div>
            <button onclick="analyze()">🔍 Analyze Image</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>

            <div id="result" class="result"></div>
        </div>

        <div class="card">
            <h3 style="color: white;">📡 API Endpoints for Flutter</h3>
            <div class="endpoints">
                <code><span class="badge">POST</span> /predict</code><br>
                <small>Send image (multipart/form-data) → Receive JSON result</small>
                <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
                <code><span class="badge">GET</span> /Working</code><br>
                <small>Check API status</small>
                <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
                <code><span class="badge">GET</span> /model/info</code><br>
                <small>Get model parameters</small>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const resultDiv = document.getElementById('result');

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    preview.src = event.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        async function analyze() {
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    resultDiv.innerHTML = `<div style="color: #ff4444;">❌ ${data.error}</div>`;
                    resultDiv.style.display = 'block';
                    return;
                }

                const isStroke = data.result === 'STROKE';
                resultDiv.className = `result ${isStroke ? 'stroke' : 'no-stroke'}`;

                let visualizationHtml = '';
                if (data.visualization) {
                    visualizationHtml = `
                        <div class="visualization">
                            <h3>🔍 Face Analysis with Landmarks</h3>
                            <img src="data:image/png;base64,${data.visualization}" alt="Face Analysis">
                        </div>
                    `;
                }

                resultDiv.innerHTML = `
                    ${visualizationHtml}
                    <h2 style="color: ${isStroke ? '#ff8888' : '#88ff88'};">${isStroke ? '⚠️ STROKE DETECTED' : '✅ NO STROKE'}</h2>
                    <p><strong>Confidence:</strong> ${data.confidence}%</p>
                    <p><strong>Stroke Score:</strong> ${data.stroke_score}/10</p>
                    <p><strong>Reason:</strong> ${data.reason}</p>
                    <div class="metrics">
                        <div class="metric"><strong>Mouth Vertical:</strong> ${data.metrics.mouth_vertical_px}px</div>
                        <div class="metric"><strong>Mouth Angle:</strong> ${data.metrics.mouth_angle_deg}°</div>
                        <div class="metric"><strong>Eye Asymmetry:</strong> ${data.metrics.eye_asymmetry_px}px</div>
                        <div class="metric"><strong>Inner Eye:</strong> ${data.metrics.inner_eye_px}px</div>
                        <div class="metric"><strong>Outer Eye:</strong> ${data.metrics.outer_eye_px}px</div>
                        <div class="metric"><strong>Eyebrow:</strong> ${data.metrics.eyebrow_px}px</div>
                    </div>
                    ${isStroke ? '<p style="color: #ff8888; margin-top: 15px;">🚨 SEEK MEDICAL ATTENTION IMMEDIATELY!</p>' : ''}
                `;
                resultDiv.style.display = 'block';

            } catch (error) {
                loading.style.display = 'none';
                resultDiv.innerHTML = `<div style="color: #ff4444;">❌ Error: ${error.message}</div>`;
                resultDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Test page for client"""
    return render_template_string(TEST_PAGE)


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for Flutter"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        result = analyze_facial_symmetry_from_image(img)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/Working', methods=['GET'])
def Working():
    """Health check endpoint"""
    return jsonify({
        "status": "Working",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Model parameters endpoint"""
    return jsonify({
        "model_name": "Stroke Detection Model",
        "version": "1.0.0",
        "parameters": {
            "mouth_threshold_small": 2,
            "mouth_threshold_medium": 3,
            "mouth_threshold_large": 4,
            "eye_threshold": 3,
            "angle_threshold": 3.0,
            "stroke_score_threshold": 2
        },
        "weights": {
            "mouth_vertical": 4,
            "mouth_angle": 2,
            "eye_asymmetry": 2,
            "inner_eye": 1,
            "outer_eye": 1,
            "eyebrow": 0.5
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 STROKE DETECTION API - PRODUCTION READY")
    print("=" * 70)
    print("\n📍 FOR FLUTTER TEAM:")
    print("   POST http://localhost:5000/predict")
    print("   Send image as multipart/form-data with key 'image'")
    print("\n📍 FOR CLIENT TESTING:")
    print("   Open in browser: http://localhost:5000")
    print("\n📍 API Endpoints:")
    print("   GET  /Working     - Working check")
    print("   GET  /model/info  - Model parameters")
    print("   POST /predict     - Stroke prediction")
    print("=" * 70)
    print("\n⚠️  Make sure both client and Flutter team use the same IP")
    print("   On same network: http://YOUR_IP:5000")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=False)