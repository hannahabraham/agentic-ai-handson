from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


class PushNotification(BaseModel):
    """A message sent to the user"""
    message: str = Field(..., description="The message sent to the user")

class PushNotificationTool(Basetool):
    name: str ="Send a push notification"
    descripton: str = (
        "This tool is used to send a push notification to the user."
    )
    args_schema: Type[BaseModel] = PushNotification

    def _run(self,message:str) ->str:
        pushover_usr = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"

        print(f" Push: {message}")
        payload = {"user":pushover_usr, "token":pushover_token, "message":message}
        requests.post(pushover_url, data=payload)
        return '{"notification":"ok"}'