pipeline {
    agent any

    environment {
        KAGGLE_CONFIG_DIR = 'C:\\WINDOWS\\system32\\config\\systemprofile\\.kaggle'
        PYTHON = 'C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
        PIP = 'C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\pip.exe'
        KAGGLE = 'C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\kaggle.exe'
    }

    stages {
        stage('Clone Repo') {
            steps {
                echo '📥 Cloning GitHub repository...'
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python packages...'
                bat '''
                    %PIP% install --upgrade pip
                    %PIP% install -r requirements.txt
                    %PIP% install werkzeug flask pytest pytest-flask
                    %PIP% list
                '''
            }
        }

        stage('Download Dataset from Kaggle') {
            steps {
                script {
                    echo '🔐 Configuring Kaggle and downloading dataset...'
                    bat '''
                        if not exist "%KAGGLE_CONFIG_DIR%" mkdir "%KAGGLE_CONFIG_DIR%"
                        copy /Y "kaggle.json" "%KAGGLE_CONFIG_DIR%\\kaggle.json"
                        %KAGGLE% datasets download -d blastchar/telco-customer-churn -p . --unzip
                    '''
                }
            }
        }

        stage('Run Model Script') {
            steps {
                echo '🚀 Running churn prediction script...'
                bat '%PYTHON% churn.py'
            }
        }

        stage('Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                bat '%PYTHON% -m pytest tests'
            }
        }

        stage('Deploy (optional)') {
            steps {
                echo '🚢 Add deployment logic here.'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for issues.'
        }
    }
}
