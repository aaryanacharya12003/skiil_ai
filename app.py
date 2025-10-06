
import os
import json
import random
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries with proper error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("Google Generative AI library imported successfully")
except ImportError as e:
    GEMINI_AVAILABLE = False
    genai = None
    logger.error(f"Failed to import google.generativeai: {e}")
    logger.error("Please install google-generativeai: pip install google-generativeai")

# Try to import document processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available. Install with: pip install python-docx")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['JSON_DATA_FOLDER'] = 'json_data'

# Get Gemini API key from environment variable (more secure)
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Configure Gemini API with better error handling
model = None
MODEL_NAME = 'gemini-flash-latest'  # Using flash latest model (closest to 1.5 flash)
if GEMINI_AVAILABLE and app.config['GEMINI_API_KEY']:
    try:
        genai.configure(api_key=app.config['GEMINI_API_KEY'])
        
        # Use gemini-flash-latest (closest to 1.5 flash functionality)
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            # Test the model with a simple prompt to verify it's working
            test_response = model.generate_content("Say 'Hello' to test the API connection.")
            logger.info(f"Gemini API configured successfully with model '{MODEL_NAME}'!")
            logger.info(f"Test response: {test_response.text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API with '{MODEL_NAME}'. Error: {e}")
            model = None
                
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        model = None
else:
    if not GEMINI_AVAILABLE:
        logger.error("Gemini library not available")
    if not app.config['GEMINI_API_KEY']:
        logger.error("No Gemini API key provided. Set GEMINI_API_KEY environment variable")

# Ensure upload and data folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['JSON_DATA_FOLDER'], exist_ok=True)

# JSON Data Management Functions
def load_json_data(filename):
    """Load data from JSON file"""
    try:
        filepath = os.path.join(app.config['JSON_DATA_FOLDER'], filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading JSON data from {filename}: {e}")
        return []

def save_json_data(filename, data):
    """Save data to JSON file"""
    try:
        filepath = os.path.join(app.config['JSON_DATA_FOLDER'], filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data to {filename}: {e}")
        return False

def get_next_id(data_list):
    """Get next available ID"""
    if not data_list:
        return 1
    return max(item.get('id', 0) for item in data_list) + 1

def init_json_storage():
    """Initialize JSON storage with dynamic project generation"""
    # Load existing projects or create dynamic ones
    projects = load_json_data('projects.json')
    
    if not projects and model:
        try:
            logger.info("Generating dynamic projects using Gemini AI...")
            # Generate diverse project ideas using AI
            prompt = """
            Generate 8 diverse software development projects. Include cutting-edge technologies and real-world applications.
            Return ONLY a JSON array in this exact format (no other text):
            [
                {
                    "id": 1,
                    "name": "Project Name",
                    "description": "Detailed description of the project including key features and objectives",
                    "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"]
                }
            ]
            
            Make projects diverse across: web development, mobile apps, AI/ML, cloud computing, data science, DevOps, cybersecurity, and IoT.
            Each project should require 4-8 relevant skills and have a detailed, professional description.
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean the response to extract only JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            try:
                projects = json.loads(response_text)
                if isinstance(projects, list) and len(projects) > 0:
                    save_json_data('projects.json', projects)
                    logger.info(f"Successfully generated {len(projects)} dynamic projects")
                else:
                    raise ValueError("Generated projects is not a valid list")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse generated projects JSON: {e}")
                logger.error(f"Raw response: {response_text}")
                projects = []
                
        except Exception as e:
            logger.error(f"Failed to generate dynamic projects: {e}")
            projects = []
    
    # Fallback to sample projects if AI generation fails or no projects exist
    if not projects:
        logger.info("Using fallback sample projects")
        sample_projects = [
            {
                "id": 1,
                "name": "AI-Powered Web Application",
                "description": "Build a modern web application with AI features including chatbot integration, user authentication, and real-time dashboard. Perfect for full-stack developers looking to integrate AI capabilities.",
                "required_skills": ["Python", "JavaScript", "Flask", "React", "AI", "Machine Learning", "SQL", "API Development"]
            },
            {
                "id": 2,
                "name": "Data Science & Analytics Platform",
                "description": "Create comprehensive data analytics platform with visualization dashboards, predictive modeling, and automated reporting. Ideal for data professionals and analysts.",
                "required_skills": ["Python", "Data Analysis", "SQL", "Pandas", "NumPy", "Matplotlib", "Tableau", "Statistics", "Machine Learning"]
            },
            {
                "id": 3,
                "name": "Mobile App Development Project",
                "description": "Develop cross-platform mobile applications using React Native with features like user authentication, push notifications, and offline functionality.",
                "required_skills": ["React Native", "JavaScript", "Mobile Development", "UI/UX", "Firebase", "API Integration"]
            },
            {
                "id": 4,
                "name": "Cloud-Native Microservices",
                "description": "Design and implement scalable microservices architecture on cloud platforms with containerization, API gateways, and monitoring systems.",
                "required_skills": ["Java", "Spring Boot", "Docker", "Kubernetes", "AWS", "Microservices", "DevOps", "API Development"]
            },
            {
                "id": 5,
                "name": "E-commerce Platform Development",
                "description": "Build full-featured online shopping platform with payment integration, inventory management, user reviews, and admin dashboard.",
                "required_skills": ["Java", "Spring", "MySQL", "Payment Integration", "Security", "REST API", "Frontend Development"]
            },
            {
                "id": 6,
                "name": "DevOps & Infrastructure Automation",
                "description": "Implement comprehensive CI/CD pipelines, infrastructure as code, monitoring, and automated deployment strategies for modern applications.",
                "required_skills": ["Docker", "Kubernetes", "AWS", "DevOps", "Linux", "Automation", "Jenkins", "Terraform", "Monitoring"]
            }
        ]
        save_json_data('projects.json', sample_projects)
    
    # Initialize empty users and skills files if they don't exist
    if not os.path.exists(os.path.join(app.config['JSON_DATA_FOLDER'], 'users.json')):
        save_json_data('users.json', [])
    if not os.path.exists(os.path.join(app.config['JSON_DATA_FOLDER'], 'user_skills.json')):
        save_json_data('user_skills.json', [])

# Initialize JSON storage on app startup
init_json_storage()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'doc', 'docx', 'txt'}

def extract_text_from_file(filepath):
    """Extract text from uploaded files"""
    try:
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf' and PDF_AVAILABLE:
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = " ".join(page.extract_text() or '' for page in reader.pages)
                return text.strip()
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
                return ""
                
        elif ext in ['.doc', '.docx'] and DOCX_AVAILABLE:
            try:
                doc = docx.Document(filepath)
                text = " ".join([p.text for p in doc.paragraphs])
                return text.strip()
            except Exception as e:
                logger.error(f"Error reading Word document: {e}")
                return ""
                
        elif ext == '.txt':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        return ""

def extract_skills_with_gemini(cv_text):
    """Extract skills from CV text using Gemini API"""
    
    def extract_skills_fallback(text):
        """Enhanced fallback method to extract skills from text"""
        logger.info("Using fallback skill extraction method")
        
        # Comprehensive skill categories
        technical_skills = {
            'python', 'java', 'javascript', 'typescript', 'html', 'css', 'sql', 'react', 'angular',
            'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'hibernate',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd', 'jenkins',
            'terraform', 'ansible', 'mongodb', 'postgresql', 'mysql', 'redis',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'
        }
        
        programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'swift', 'kotlin', 'r', 'scala', 'perl', 'shell', 'bash', 'powershell'
        }
        
        frameworks_tools = {
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'hibernate', 'docker', 'kubernetes', 'jenkins', 'terraform',
            'ansible', 'numpy', 'pandas', 'tensorflow', 'pytorch', 'scikit-learn',
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch'
        }
        
        soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving',
            'analytical thinking', 'creative', 'time management', 'project management',
            'collaboration', 'adaptability', 'organization', 'critical thinking',
            'presentation', 'negotiation', 'mentoring', 'training'
        }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        import re
        
        # Find all matches using word boundaries
        found_tech = []
        found_prog = []
        found_frame = []
        found_soft = []
        
        for skill in technical_skills:
            if re.search(r'\b' + re.escape(skill.replace('.', r'\.')) + r'\b', text_lower):
                found_tech.append(skill.title())
        
        for lang in programming_languages:
            if re.search(r'\b' + re.escape(lang.replace('.', r'\.')) + r'\b', text_lower):
                found_prog.append(lang.title())
        
        for tool in frameworks_tools:
            if re.search(r'\b' + re.escape(tool.replace('.', r'\.')) + r'\b', text_lower):
                found_frame.append(tool.title())
        
        for skill in soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_soft.append(skill.title())
        
        # Build result
        result = []
        if found_tech:
            result.append({"Skill Category": "Technical Skills", "Skills": list(set(found_tech))[:10]})
        if found_prog:
            result.append({"Skill Category": "Programming Languages", "Skills": list(set(found_prog))[:8]})
        if found_frame:
            result.append({"Skill Category": "Frameworks & Tools", "Skills": list(set(found_frame))[:10]})
        if found_soft:
            result.append({"Skill Category": "Soft Skills", "Skills": list(set(found_soft))[:8]})
        
        # Ensure we have at least some skills
        if not result:
            result = [
                {"Skill Category": "Technical Skills", "Skills": ["Data Analysis", "Problem Solving", "Research"]},
                {"Skill Category": "Soft Skills", "Skills": ["Communication", "Team Collaboration", "Adaptability"]}
            ]
        
        return result
    
    # Try Gemini API first
    if model:
        try:
            logger.info("Using Gemini AI for skill extraction")
            
            # Enhanced prompt for better skill extraction
            prompt = f"""
            As an expert HR analyst, analyze this CV content and extract ALL relevant skills comprehensively.
            
            CV Content:
            {cv_text[:4000]}
            
            Extract and categorize ALL skills mentioned or implied from:
            - Technical skills and programming languages
            - Frameworks, tools, and technologies  
            - Soft skills and interpersonal abilities
            - Domain expertise and certifications
            - Skills implied from job roles, projects, and achievements
            
            Return ONLY a JSON array in this exact format:
            [
                {{"Skill Category": "Programming Languages", "Skills": ["Python", "Java", "JavaScript"]}},
                {{"Skill Category": "Technical Skills", "Skills": ["Machine Learning", "Data Analysis", "API Development"]}},
                {{"Skill Category": "Frameworks & Tools", "Skills": ["React", "Django", "Docker"]}},
                {{"Skill Category": "Soft Skills", "Skills": ["Leadership", "Communication", "Problem Solving"]}},
                {{"Skill Category": "Domain Expertise", "Skills": ["Healthcare", "Finance", "E-commerce"]}}
            ]
            
            Guidelines:
            1. Be comprehensive - extract ALL skills mentioned
            2. Use standard industry terminology
            3. Categorize appropriately 
            4. Avoid duplicates within categories
            5. Include 3-10 skills per category
            6. Return ONLY the JSON array, no other text
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean the response to extract only JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            skills_json = json.loads(response_text)
            
            # Validate structure
            if isinstance(skills_json, list) and len(skills_json) > 0:
                valid = True
                for category in skills_json:
                    if not isinstance(category, dict) or 'Skill Category' not in category or 'Skills' not in category:
                        valid = False
                        break
                
                if valid:
                    logger.info(f"Successfully extracted {len(skills_json)} skill categories using Gemini")
                    return skills_json
                else:
                    logger.warning("Invalid skill structure from Gemini, using fallback")
            else:
                logger.warning("Invalid response format from Gemini, using fallback")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error from Gemini response: {e}")
            logger.error(f"Raw response: {response_text[:200]}...")
        except Exception as e:
            logger.error(f"Gemini API error during skill extraction: {e}")
    
    # Use fallback method if Gemini fails or is not available
    return extract_skills_fallback(cv_text)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """API status endpoint"""
    return jsonify({
        'gemini_available': GEMINI_AVAILABLE,
        'model_configured': model is not None,
        'model_name': MODEL_NAME if model else None,
        'pdf_support': PDF_AVAILABLE,
        'docx_support': DOCX_AVAILABLE,
        'skill_gap_analysis': True
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Handle CV upload and skill extraction"""
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        experience_level = request.form.get('experience_level')
        file = request.files.get('cv_file')
        
        # Validate inputs
        if not all([name, email, experience_level]):
            return jsonify({"error": "Missing required fields"}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload PDF, DOC, DOCX, or TXT file"}), 400
        
        # Save file
        filename = secure_filename(str(file.filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Extract text from CV
        text = extract_text_from_file(filepath)
        if not text or len(text.strip()) < 50:
            return jsonify({"error": "Could not extract sufficient text from CV. Please check the file format."}), 400
        
        logger.info(f"Extracted {len(text)} characters from CV")
        
        # Extract skills using Gemini API
        skills_json = extract_skills_with_gemini(text)
        
        # Load existing users and get next ID
        users = load_json_data('users.json')
        user_id = get_next_id(users)
        
        # Create user record
        user_record = {
            "id": user_id,
            "name": name,
            "email": email,
            "experience_level": experience_level,
            "upload_date": datetime.now().isoformat(),
            "cv_filename": filename
        }
        
        # Add user to users list and save
        users.append(user_record)
        if not save_json_data('users.json', users):
            return jsonify({"error": "Failed to save user data"}), 500
        
        # Save user skills
        user_skills_record = {
            "user_id": user_id,
            "user_name": name,
            "skills_data": skills_json,
            "extraction_date": datetime.now().isoformat()
        }
        
        # Load existing user skills and add new record
        all_user_skills = load_json_data('user_skills.json')
        all_user_skills.append(user_skills_record)
        if not save_json_data('user_skills.json', all_user_skills):
            return jsonify({"error": "Failed to save user skills"}), 500
        
        # Also save individual user's skills in separate file for easy access
        user_skills_filename = f'user_{user_id}_skills.json'
        if not save_json_data(user_skills_filename, skills_json):
            logger.warning(f"Failed to save individual user skills file: {user_skills_filename}")
        
        logger.info(f"Successfully processed CV for user {user_id}: {name}")
        
        # Return skills with user ID for recommendations
        return jsonify({
            "success": True,
            "skills": skills_json, 
            "user_id": user_id,
            "message": f"CV processed successfully. Extracted {sum(len(cat['Skills']) for cat in skills_json)} skills."
        })
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({"error": f"An error occurred while processing your CV: {str(e)}"}), 500

@app.route('/recommendations/<int:user_id>')
def recommendations(user_id):
    """Get project recommendations for a user with skill gap analysis"""
    try:
        # Load user data
        users = load_json_data('users.json')
        user = next((u for u in users if u['id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Load user skills
        user_skills_data = load_json_data(f'user_{user_id}_skills.json')
        if not user_skills_data:
            return jsonify({'error': 'No skills found for user'}), 404
        
        # Extract all skills from all categories
        user_skills = []
        for category in user_skills_data:
            user_skills.extend(category.get('Skills', []))
        
        # Load projects
        projects = load_json_data('projects.json')
        if not projects:
            return jsonify({'error': 'No projects available'}), 404
        
        # AI-powered matching with skills gap analysis using Gemini
        matched = []
        if model:
            try:
                logger.info(f"Generating AI recommendations with skill gap analysis for user {user_id}")
                
                # Create detailed project descriptions for AI
                project_details = []
                for proj in projects:
                    skills_str = ', '.join(proj['required_skills'])
                    project_details.append(f"ID: {proj['id']}\nProject: {proj['name']}\nDescription: {proj['description']}\nRequired Skills: {skills_str}")
                
                prompt = f"""
                You are an expert career advisor and technical mentor. Analyze the candidate's skills and provide comprehensive project recommendations with detailed skill gap analysis.
                
                CANDIDATE PROFILE:
                - Name: {user['name']}
                - Experience Level: {user['experience_level']}
                - Current Skills: {', '.join(user_skills)}
                
                AVAILABLE PROJECTS:
                {chr(10).join(project_details)}
                
                TASK: 
                1. Recommend the TOP 4-6 most suitable projects based on skill match and career growth potential
                2. For each recommended project, provide detailed skill gap analysis
                3. Suggest specific skills to improve and learn
                4. Provide learning recommendations and next steps
                
                Consider:
                - Current skill match percentage
                - Experience level appropriateness
                - Skills that are missing but learnable
                - Career progression opportunities
                - Growth potential and learning curve
                
                Return ONLY a valid JSON array with this exact structure (no additional text or formatting):
                [
                    {{
                        "project_id": 1,
                        "match_score": 85,
                        "matching_skills": ["Python", "API Development"],
                        "missing_skills": ["Docker", "Kubernetes"],
                        "skills_to_improve": ["Advanced Python", "System Design"],
                        "learning_recommendations": [
                            "Complete Docker fundamentals course",
                            "Practice Kubernetes basics with minikube",
                            "Build 2-3 microservices projects"
                        ],
                        "difficulty_level": "Intermediate",
                        "estimated_learning_time": "2-3 months",
                        "why_recommended": "Perfect next step to advance your backend development skills",
                        "career_benefits": ["DevOps skills", "Cloud-native development", "Scalable architecture"]
                    }}
                ]
                
                IMPORTANT: Return ONLY the JSON array, no other text. Use project_id numbers from the available projects list above.
                """
                
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                logger.info(f"Raw AI response: {response_text[:200]}...")
                
                # Clean the response to extract only JSON
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                # Remove any leading/trailing text that's not JSON
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
                
                logger.info(f"Cleaned JSON: {response_text[:200]}...")
                
                try:
                    ai_recommendations = json.loads(response_text)
                    logger.info(f"Successfully parsed {len(ai_recommendations)} AI recommendations")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"Problematic JSON: {response_text}")
                    ai_recommendations = []
                
                # Match AI recommendations with actual projects
                if ai_recommendations:
                    for ai_rec in ai_recommendations:
                        project_id = ai_rec.get('project_id')
                        project = next((p for p in projects if p['id'] == project_id), None)
                        
                        if project:
                            # Enhanced project data with skill gap analysis
                            enhanced_project = {
                                'id': project['id'],
                                'name': project['name'],
                                'description': project['description'],
                                'required_skills': project['required_skills'],
                                'match_score': ai_rec.get('match_score', 0),
                                'matching_skills': ai_rec.get('matching_skills', []),
                                'missing_skills': ai_rec.get('missing_skills', []),
                                'skills_to_improve': ai_rec.get('skills_to_improve', []),
                                'learning_recommendations': ai_rec.get('learning_recommendations', []),
                                'difficulty_level': ai_rec.get('difficulty_level', 'Unknown'),
                                'estimated_learning_time': ai_rec.get('estimated_learning_time', 'Unknown'),
                                'why_recommended': ai_rec.get('why_recommended', ''),
                                'career_benefits': ai_rec.get('career_benefits', [])
                            }
                            matched.append(enhanced_project)
                    
                    if matched:
                        matched.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                        logger.info(f"Generated {len(matched)} AI recommendations with skill gap analysis")
                
            except Exception as e:
                logger.error(f"AI recommendation error: {e}")
                matched = []
        
        # Enhanced fallback matching with basic skill gap analysis
        if not matched:
            logger.info("Using fallback recommendation algorithm with basic skill gap analysis")
            scored_projects = []
            for proj in projects:
                # Identify matching and missing skills
                matching_skills = []
                missing_skills = []
                
                for req_skill in proj['required_skills']:
                    req_lower = req_skill.lower().strip()
                    is_match = any(user_skill.lower() in req_lower or req_lower in user_skill.lower() 
                                 for user_skill in user_skills)
                    if is_match:
                        matching_skills.append(req_skill)
                    else:
                        missing_skills.append(req_skill)
                
                # Calculate match score
                match_count = len(matching_skills)
                total_required = len(proj['required_skills'])
                
                if total_required > 0:
                    match_percentage = (match_count / total_required) * 100
                else:
                    match_percentage = 0
                
                # Experience level bonus
                experience_bonus = 0
                desc_lower = proj['description'].lower()
                if user['experience_level'] == 'Junior' and any(word in desc_lower for word in ['beginner', 'entry', 'junior', 'basic']):
                    experience_bonus = 15
                elif user['experience_level'] == 'Mid' and any(word in desc_lower for word in ['intermediate', 'mid', 'moderate']):
                    experience_bonus = 15
                elif user['experience_level'] == 'Senior' and any(word in desc_lower for word in ['advanced', 'senior', 'expert', 'lead']):
                    experience_bonus = 15
                
                final_score = min(match_percentage + experience_bonus, 100)
                
                if match_count > 0 or len(missing_skills) <= 4:  # Include projects with some skills or few missing skills
                    # Basic learning recommendations
                    basic_recommendations = []
                    if missing_skills:
                        basic_recommendations.extend([f"Learn {skill}" for skill in missing_skills[:3]])
                    if len(matching_skills) > 0:
                        basic_recommendations.append(f"Strengthen your {matching_skills[0]} skills")
                    
                    # Skills to improve (existing skills that could be enhanced)
                    skills_to_improve = []
                    for match_skill in matching_skills[:2]:
                        skills_to_improve.append(f"Advanced {match_skill}")
                    
                    scored_projects.append({
                        'id': proj['id'],
                        'name': proj['name'],
                        'description': proj['description'],
                        'required_skills': proj['required_skills'],
                        'match_score': round(final_score, 1),
                        'matching_skills': matching_skills,
                        'missing_skills': missing_skills,
                        'skills_to_improve': skills_to_improve,
                        'learning_recommendations': basic_recommendations,
                        'difficulty_level': f'Suitable for {user["experience_level"]} level',
                        'estimated_learning_time': f"{len(missing_skills) * 2}-{len(missing_skills) * 4} weeks" if missing_skills else "Ready to start",
                        'why_recommended': f"Matches {match_count}/{total_required} required skills. Good for skill development.",
                        'career_benefits': ['Skill development', 'Project experience', 'Portfolio building']
                    })
            
            # Sort by score and take top matches
            scored_projects.sort(key=lambda x: x['match_score'], reverse=True)
            matched = scored_projects[:6]
            
            logger.info(f"Generated {len(matched)} fallback recommendations with skill gap analysis")
        
        logger.info(f"Returning {len(matched)} recommendations with skill gap analysis for user {user_id}")
        
        return jsonify({
            'user': user,
            'user_skills': user_skills,
            'user_skills_categories': user_skills_data,
            'recommendations': matched,
            'total_skills': len(user_skills),
            'ai_powered': model is not None,
            'skill_gap_analysis': True
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        return jsonify({'error': f"Failed to generate recommendations: {str(e)}"}), 500

@app.route('/data')
def view_data():
    """View all stored JSON data"""
    try:
        users = load_json_data('users.json')
        user_skills = load_json_data('user_skills.json')
        projects = load_json_data('projects.json')
        
        return jsonify({
            'users': users,
            'user_skills': user_skills,
            'projects': projects,
            'total_users': len(users),
            'total_projects': len(projects),
            'system_status': {
                'gemini_available': GEMINI_AVAILABLE,
                'model_configured': model is not None,
                'pdf_support': PDF_AVAILABLE,
                'docx_support': DOCX_AVAILABLE
            }
        })
    except Exception as e:
        logger.error(f"Error in data endpoint: {e}")
        return jsonify({'error': f"Failed to load data: {str(e)}"}), 500

@app.route('/user_skills/<int:user_id>')
def get_user_skills(user_id):
    """Get skills for a specific user in the exact format requested"""
    try:
        user_skills = load_json_data(f'user_{user_id}_skills.json')
        if not user_skills:
            return jsonify({'error': 'No skills found for this user'}), 404
        
        return jsonify({
            'user_id': user_id,
            'skills': user_skills,
            'total_categories': len(user_skills),
            'total_skills': sum(len(cat.get('Skills', [])) for cat in user_skills)
        })
    except Exception as e:
        logger.error(f"Error getting user skills: {e}")
        return jsonify({'error': f"Failed to get user skills: {str(e)}"}), 500

if __name__ == '__main__':
    # Print startup information
    print("\n" + "="*50)
    print("CV SKILL EXTRACTOR & PROJECT RECOMMENDER")
    print("="*50)
    print(f"Gemini API Available: {GEMINI_AVAILABLE}")
    print(f"Model Configured: {model is not None}")
    if model:
        print(f"Model: {MODEL_NAME}")
    print(f"PDF Support: {PDF_AVAILABLE}")
    print(f"DOCX Support: {DOCX_AVAILABLE}")
    print(f"Skill Gap Analysis: Enabled")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
