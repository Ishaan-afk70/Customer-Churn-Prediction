pipeline {
    agent any

    environment {
        PYTHON_PATH = "C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
    }

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat "${env.PYTHON_PATH} -m pip install -r requirements.txt || ${env.PYTHON_PATH} -m pip install pandas scikit-learn matplotlib pytest kaggle"
            }
        }

        stage('Setup Kaggle API Key') {
            steps {
                withCredentials([file(credentialsId: 'kaggle-json', variable: 'KAGGLE_SECRET')]) {
                    bat '''
                    mkdir "%USERPROFILE%\\.kaggle"
                    copy "%KAGGLE_SECRET%" "%USERPROFILE%\\.kaggle\\kaggle.json"
                    '''
                }
            }
        }

        stage('Download Dataset from Kaggle') {
            steps {
                bat '''
                kaggle datasets download -d blastchar/telco-customer-churn -p . --unzip
                '''
            }
        }

        stage('Run Model Script') {
            steps {
                bat "${env.PYTHON_PATH} churn.py"
            }
        }

        stage('Run Tests') {
            steps {
                bat "${env.PYTHON_PATH} -m pytest tests"
            }
        }

        stage('Deploy (optional)') {
            steps {
                echo '🚀 You can add deployment logic here.'
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
