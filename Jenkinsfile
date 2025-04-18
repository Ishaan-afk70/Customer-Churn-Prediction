pipeline {
    agent any

    environment {
        // Set the environment variable for Kaggle CLI authentication
        KAGGLE_CONFIG_DIR = 'C:\\WINDOWS\\system32\\config\\systemprofile\\.kaggle'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '"C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pip install -r requirements.txt || "C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pip install pandas scikit-learn matplotlib pytest kaggle'
            }
        }

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
                bat '"C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" churn.py'
            }
        }

        stage('Run Tests') {
            steps {
                bat '"C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m pytest tests'
            }
        }

        stage('Deploy (optional)') {
            steps {
                echo 'You can add deployment commands here (e.g., run container, push to Docker Hub)'
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
