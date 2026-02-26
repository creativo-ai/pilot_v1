from agent_router import EmailAgent
agent = EmailAgent()
user_id = "manar"
brand_id = "creativo"
user_input= input()
conversation_history=[]
response = agent.handle(user_input, conversation_history, user_id, brand_id)