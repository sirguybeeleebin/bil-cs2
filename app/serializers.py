from rest_framework import serializers

class PredictSerializer(serializers.Serializer):
    map_id = serializers.IntegerField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    team1_player1_id = serializers.IntegerField()
    team1_player2_id = serializers.IntegerField()
    team1_player3_id = serializers.IntegerField()
    team1_player4_id = serializers.IntegerField()
    team1_player5_id = serializers.IntegerField()
    team2_player1_id = serializers.IntegerField()
    team2_player2_id = serializers.IntegerField()
    team2_player3_id = serializers.IntegerField()
    team2_player4_id = serializers.IntegerField()
    team2_player5_id = serializers.IntegerField()
    team_id_start_ct = serializers.IntegerField()
