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
    """Real MCP Client integrating with Google Workspace APIs."""
    
    def __init__(self):
        self.creds = self._get_credentials()
        self.docs_service = build('docs', 'v1', credentials=self.creds)
        self.gmail_service = build('gmail', 'v1', credentials=self.creds)

    def _get_credentials(self):
        creds = None
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Requires credentials.json from Google Cloud Console
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        return creds
        
    def draft_google_doc(self, title: str, content: str) -> str:
        """Creates a Google Doc and inserts content, returning the URL."""
        # Create empty doc
        doc = self.docs_service.documents().create(body={'title': title}).execute()
        doc_id = doc.get('documentId')
        
        # Insert text
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
        message = EmailMessage()
        message.set_content(body)
        message['To'] = 'customer@example.com' # Placeholder
        message['From'] = 'me'
        message['Subject'] = subject
        
        # Base64 encode the message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        create_message = {
            'message': {
                'raw': encoded_message
            }
        }
        
        draft = self.gmail_service.users().drafts().create(
            userId="me", body=create_message).execute()
            
        draft_id = draft['message']['id']
        return f"https://mail.google.com/mail/u/0/#drafts/{draft_id}"
