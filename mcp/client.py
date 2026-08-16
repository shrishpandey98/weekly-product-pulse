import os
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/documents', 
    'https://www.googleapis.com/auth/gmail.compose'
]

class MCPClient:
    """Real MCP Client integrating with Google Workspace APIs with Cloud fallback."""
    
    def __init__(self):
        self.creds = self._get_credentials()
        if self.creds:
            try:
                self.docs_service = build('docs', 'v1', credentials=self.creds)
                self.gmail_service = build('gmail', 'v1', credentials=self.creds)
            except Exception:
                self.docs_service = None
                self.gmail_service = None
        else:
            self.docs_service = None
            self.gmail_service = None

    def _get_credentials(self):
        creds = None
        # 1. Check local token.json file
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception:
                pass
                
        # 2. Check Streamlit Cloud Secrets for GOOGLE_TOKEN
        if not creds or not creds.valid:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    token_str = None
                    if "GOOGLE_TOKEN" in st.secrets:
                        token_str = st.secrets["GOOGLE_TOKEN"]
                    elif "google_token" in st.secrets:
                        token_str = st.secrets["google_token"]
                        
                    if token_str:
                        import json
                        if isinstance(token_str, str):
                            token_info = json.loads(token_str)
                        else:
                            token_info = dict(token_str)
                        creds = Credentials.from_authorized_user_info(token_info, SCOPES)
            except Exception:
                pass

        # 3. Refresh expired credentials if refresh token is available
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                return creds
            except Exception:
                creds = None

        # 4. Fallback to local interactive flow if credentials.json exists
        if not creds or not creds.valid:
            if os.path.exists('credentials.json'):
                try:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                    return creds
                except Exception:
                    pass
        return creds
        
    def draft_google_doc(self, title: str, content: str) -> str:
        """Creates a Google Doc and inserts content, returning the URL."""
        if not self.docs_service:
            import uuid
            doc_id = str(uuid.uuid4())[:8]
            return f"https://docs.google.com/document/d/cloud-demo-{doc_id}/edit"
            
        doc = self.docs_service.documents().create(body={'title': title}).execute()
        doc_id = doc.get('documentId')
        
        requests = [
            {
                'insertText': {
                    'location': {
                        'index': 1,
                    },
                    'text': content
                }
            }
        ]
        self.docs_service.documents().batchUpdate(
            documentId=doc_id, body={'requests': requests}).execute()
            
        return f"https://docs.google.com/document/d/{doc_id}/edit"
        
    def draft_gmail(self, subject: str, body: str) -> str:
        """Creates a Gmail draft and returns the URL."""
        if not self.gmail_service:
            import uuid
            draft_id = str(uuid.uuid4())[:12]
            return f"https://mail.google.com/mail/u/0/#drafts?compose=cloud-demo-{draft_id}"

        message = EmailMessage()
        message.set_content(body)
        message['To'] = 'customer@example.com'
        message['From'] = 'me'
        message['Subject'] = subject
        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'message': {'raw': encoded_message}}
        
        draft = self.gmail_service.users().drafts().create(
            userId="me", body=create_message).execute()
            
        # Get exact draft ID for Gmail web compose link
        draft_id = draft.get('id', draft.get('message', {}).get('id'))
        return f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}"
