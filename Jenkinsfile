pipeline {
    agent any

    environment {
        // Kaggle API credentials directory for Jenkins (LocalSystem)
        KAGGLE_CONFIG_DIR = 'C:\\WINDOWS\\system32\\config\\systemprofile\\.kaggle'
        PYTHON = 'C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
    }

    stages {
        stage('Clone Repo') {
            steps {
                echo '📦 Cloning the repository...'
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '📥 Installing Python dependencies...'
                bat '''
                    %PYTHON% -m pip install --upgrade pip
                    %PYTHON% -m pip install -r requirements.txt || %PYTHON% -m pip install pandas scikit-learn matplotlib pytest kaggle
                '''
            }
        }

        // ✅ Your Kaggle download stage
        stage('Download Dataset from Kaggle') {
            steps {
                script {
                    // Make sure kaggle.json is available and accessible
                    bat 'mkdir "C:\\WINDOWS\\system32\\config\\systemprofile\\.kaggle"'
                    bat 'copy "C:\\ProgramData\\Jenkins\\.jenkins\\workspace\\Customer-Churn-Testing\\kaggle.json" "C:\\WINDOWS\\system32\\config\\systemprofile\\.kaggle\\kaggle.json"'
                    
                    // Download dataset from Kaggle using Kaggle API
                    bat 'kaggle datasets download -d blastchar/telco-customer-churn -p . --unzip'
                }
            }
        }

        stage('Run Model Script') {
            steps {
                echo '🚀 Running churn.py...'
                bat '%PYTHON% churn.py'
            }
        }

        stage('Run Tests') {
            steps {
                echo '🧪 Running tests...'
                bat '%PYTHON% -m pytest tests'
            }
        }

        stage('Deploy (optional)') {
            steps {
                echo '🚢 Optional deployment stage.'
            }
        }
    }

    post {
        success {
            echo '✅ CI/CD pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for issues.'
        }
    }
}
