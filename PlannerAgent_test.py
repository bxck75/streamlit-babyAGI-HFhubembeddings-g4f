from langchain import LLMChain, PromptTemplate
from langchain.agents import Agent, AgentExecutor

class PlannerAgent(Agent):
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input="Goal: {goal}\nAvailable actions: {actions}\nCurrent state: {state}\nPlan:"
        )
        
    def run(self, goal, actions, state):
        plan = self.llm(self.prompt.format(goal=goal, actions=actions, state=state))
        return AgentExecutor.execute_plan(plan, actions)
        
llm = LLMChain() 
agent = PlannerAgent(llm)

goal = "Deliver package to 123 Main St" 
actions = ["Drive to warehouse", "Pick up package", "Drive to 123 Main St", "Deliver package"]
state = "At home"

agent.run(goal, actions, state)