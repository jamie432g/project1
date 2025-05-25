import pandas as pd

def save_speed_distance_to_csv(speed_distance_dict, output_path="output_csv/speed_distance_data.csv"):
    data = []
    
    for _, players in speed_distance_dict.items():
            for track_id, player_data in players.items():
                team = player_data["team"]
                total_distance = player_data["total_distance"]
                
                speeds = [entry["speed"] for entry in player_data["data"] if "speed" in entry]
                
                # try to limit unrealistic speeds from inconsistent homography
                if speeds:
                    min_speed = min(speeds)
                    valid_speeds = [s for s in speeds if s <= 38]
                    top_speed = max(valid_speeds) if valid_speeds else min_speed  
                else:
                    min_speed, top_speed = 0, 0  
                    
                data.append([track_id, team, min_speed, top_speed, total_distance])
    
    df = pd.DataFrame(data, columns=["Track ID", "Team", "Minimum Speed (km/h)", "Top Speed (km/h)", "Total Distance (m)"])
    df.to_csv(output_path, index=False)
    print(f"CSV saved successfully: {output_path}")

