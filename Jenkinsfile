pipeline {
    agent any

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt || pip install pandas scikit-learn matplotlib'
            }
        }

        stage('Run Model Script') {
            steps {
                bat 'python churn_prediction.py'
            }
        }
    }
}
