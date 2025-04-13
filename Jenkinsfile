pipeline {
    agent any

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'master', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt || pip install pandas scikit-learn matplotlib'
            }
        }

        stage('Run Model Script') {
            steps {
                sh 'python churn_prediction.py'
            }
        }
    }
}
