from rest_framework import serializers

from app.models import Prediction


class PredictionRequestSerializer(serializers.Serializer):
    map_id = serializers.IntegerField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()

    def validate(self, data):
        if data["team1_id"] == data["team2_id"]:
            raise serializers.ValidationError("Команды должны быть разными")
        return data


class PredictionResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ["id", "status", "result", "created_at"]
        read_only_fields = fields
