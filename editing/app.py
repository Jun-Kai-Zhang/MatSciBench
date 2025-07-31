import os
import csv
import argparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Set up argument parser
parser = argparse.ArgumentParser(description='Run the editing server with a specific CSV file')
parser.add_argument('--csv', type=str, default='datasets/ScienceAndEngineeringofMaterials/qa.csv',
                   help='Path to the CSV file containing questions')
args = parser.parse_args()

app = Flask(__name__)

# Get the current working directory for dataset base path
DATASET_BASE = os.getcwd()
DATASET_BASE = os.path.join(DATASET_BASE, 'datasets')
# Extract the dataset name from the CSV path
DATASET_NAME = args.csv.split('/')[-2]

    


# Create a path to the dataset directory
DATASET_DIR = os.path.join(DATASET_BASE, DATASET_NAME)



# Configuration
CSV_FILE = args.csv

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/<path:filepath>')
def serve_any_file(filepath):

    full_path = os.path.join(DATASET_DIR, filepath) 
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)

    # Print for debugging
    print(f"Serving file: {filepath}")
    print(f"Full path: {full_path}")
    print(f"Directory: {directory}")
    print(f"Filename: {filename}")

    
    # If you want to limit it to only subfolders or only images, add checks here.
    return send_from_directory(directory, filename)


# In-memory storage for questions
questions = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_questions():
    global questions
    questions = []
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert answer_type to type if it exists and type doesn't
            if 'answer_type' in row and 'type' not in row:
                row['type'] = row['answer_type']
                del row['answer_type']
            # Look for legacy field if type doesn't exist
            if 'type' not in row and 'problem_type' in row:
                row['type'] = row['problem_type']
            # Set default type if not present
            if 'type' not in row:
                row['type'] = 'NUM'  # Default value changed to NUM
            # Add unit field if it doesn't exist
            if 'unit' not in row:
                row['unit'] = ''  # Default empty value
            # Add notes field if it doesn't exist
            if 'notes' not in row:
                row['notes'] = ''  # Default empty value
            
            # Handle category fields
            categories = ['Materials', 'Properties', 'Structures', 
                        'Fundamental_Mechanisms', 'Processes', 'Failure_Mechanisms']
            for category in categories:
                if category not in row:
                    row[category] = ''  # Default to empty string
            
            questions.append(row)

def save_questions():
    global questions
    if questions:
        # Ensure all questions have the required fields
        for q in questions:
            if 'type' not in q:
                q['type'] = 'NUM'  # Default value changed to NUM
            if 'unit' not in q:
                q['unit'] = ''  # Default empty value
            if 'notes' not in q:
                q['notes'] = ''  # Default empty value
            
            # Ensure all category fields exist
            categories = ['Materials', 'Properties', 'Structures', 
                        'Fundamental_Mechanisms', 'Processes', 'Failure_Mechanisms']
            for category in categories:
                if category not in q:
                    q[category] = ''  # Default to empty string
            
            # Remove any old problem_type or answer_type fields
            if 'problem_type' in q:
                del q['problem_type']
            if 'answer_type' in q:
                del q['answer_type']
            # Remove old domain field if it exists
            if 'domain' in q:
                del q['domain']
                
        # Make sure type is in the fieldnames
        fieldnames = list(questions[0].keys())
        # If there's a "problem_type" in fieldnames, replace it with "type"
        if 'problem_type' in fieldnames and 'type' not in fieldnames:
            fieldnames[fieldnames.index('problem_type')] = 'type'
            
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                writer.writerow(q)

# Load questions at startup
load_questions()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions', methods=['GET'])
def get_all_questions():
    # Return a list of all question IDs (qid)
    qids = [q['qid'] for q in questions]
    return jsonify(qids)

@app.route('/question/<string:qid>', methods=['GET'])
def get_question(qid):
    for q in questions:
        if q['qid'] == qid:
            return jsonify(q)
    return jsonify({"error": "Question not found"}), 404

@app.route('/question/<string:qid>', methods=['PUT'])
def update_question(qid):
    data = request.get_json()
    updated = False
    for q in questions:
        if q['qid'] == qid:
            if 'question' in data:
                q['question'] = data['question']
            if 'solution' in data:
                q['solution'] = data['solution']
            if 'answer' in data:
                q['answer'] = data['answer']
            if 'type' in data:
                q['type'] = data['type']
            if 'number_of_answers' in data:
                q['number_of_answers'] = data['number_of_answers']
            if 'unit' in data:
                q['unit'] = data['unit']
            if 'notes' in data:
                q['notes'] = data['notes']
            updated = True
            break
    if updated:
        save_questions()
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "Question not found"}), 404

@app.route('/question/<string:qid>', methods=['DELETE'])
def delete_question(qid):
    global questions
    initial_length = len(questions)
    questions = [q for q in questions if q['qid'] != qid]
    
    if len(questions) < initial_length:
        save_questions()
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "Question not found"}), 404

# -----------------------------------------------------------
# 2) Endpoint to handle image uploads
# -----------------------------------------------------------
@app.route('/question/<string:qid>/image', methods=['POST'])
def upload_image(qid):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        images_dir = os.path.join(DATASET_BASE, DATASET_NAME, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)

        file_path = os.path.join(images_dir, filename)
        file.save(file_path)

        # Now we update the CSV record's image field by appending the new image
        updated = False
        for q in questions:
            if q['qid'] == qid:
                image_path = f"{DATASET_NAME}/images/{filename}"
                
                # If image field exists and is not empty, append to existing images
                if 'image' in q and q['image'].strip():
                    existing_images = q['image'].split(',')
                    # Avoid duplicates
                    if image_path not in existing_images:
                        existing_images.append(image_path)
                        q['image'] = ','.join(existing_images)
                else:
                    q['image'] = image_path
                
                updated = True
                break

        if updated:
            # Save to CSV file
            save_questions()
            return jsonify({"status": "ok", "filename": image_path})
        else:
            return jsonify({"error": "Question not found"}), 404
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/question/<string:qid>/image', methods=['DELETE'])
def remove_image(qid):
    data = request.get_json()
    if not data or 'image_path' not in data:
        return jsonify({"error": "No image path provided"}), 400
    
    image_path = data['image_path']
    updated = False
    
    for q in questions:
        if q['qid'] == qid and 'image' in q and q['image'].strip():
            existing_images = q['image'].split(',')
            if image_path in existing_images:
                existing_images.remove(image_path)
                q['image'] = ','.join(existing_images)
                updated = True
                break
    
    if updated:
        save_questions()
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "Image not found for this question"}), 404

@app.route('/questions', methods=['POST'])
def create_question():
    global questions
    
    # Get custom ID if provided in the request
    data = request.get_json() or {}
    custom_qid = data.get('qid')
    
    if custom_qid:
        # Check if ID already exists
        if any(q['qid'] == custom_qid for q in questions):
            return jsonify({"error": "Question ID already exists"}), 400
        new_qid = custom_qid
    else:
        # Generate a new unique ID
        if questions:
            # Extract existing IDs and find the next available number
            existing_ids = [int(q['qid'].replace('q', '')) for q in questions if q['qid'].startswith('q')]
            next_id = max(existing_ids) + 1 if existing_ids else 1
        else:
            next_id = 1
        
        new_qid = f"q{next_id}"
    
    # Create an empty question with all required fields
    new_question = {
        'qid': new_qid,
        'question': '',
        'solution': '',
        'answer': '',
        'image': '',
        'type': 'NUM',  # Default type changed to NUM
        'unit': '',  # Default empty unit
        'notes': '',  # Default empty notes
        'Materials': '',  # Default empty string
        'Properties': '',  # Default empty string
        'Structures': '',  # Default empty string
        'Fundamental_Mechanisms': '',  # Default empty string
        'Processes': '',  # Default empty string
        'Failure_Mechanisms': ''  # Default empty string
    }
    
    questions.append(new_question)
    save_questions()
    
    return jsonify({"status": "ok", "qid": new_qid})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
    print("app.root_path =", app.root_path)

