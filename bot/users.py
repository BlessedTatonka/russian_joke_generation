class UserState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.reset_user()
        
    def reset_user(self):
        self.params = {
            'temperature': 0.5,
            'k': 2           
        }
        
        
class BotState:
    def __init__(self):
        self.user_states = {}
        
    def get_user_state(self, user_id):
        if user_id not in self.user_states.keys():
            self.user_states[user_id] = UserState(user_id)
            
        return self.user_states[user_id]
            
        
       
        
    
    
    