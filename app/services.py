class PredictionService:
    def __init__(self, repo):
        self.repo = repo

    def create(self, payload):
        return self.repo.create(payload)

    def set_status(self, prediction_id, status):
        self.repo.update_status(prediction_id, status)

    def save_result(self, prediction_id, result):
        self.repo.update_result(prediction_id, result)
