import React, { useState } from 'react';
import { Container, Typography, Box, ThemeProvider, createTheme } from '@mui/material';
import UploadSection from './components/UploadSection';
import ChatBox from './components/ChatBox';
import QuerySection from './components/QuerySection';
import theme from './theme';


function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  const handleUpload = (data) => {
    setIsConnected(true);
    setChatHistory([
      ...chatHistory,
      { role: 'assistant', content: 'Database connected successfully!' },
      { role: 'assistant', content: data.description },
    ]);
  };

  const handleConnect = (data) => {
    setIsConnected(true);
    setChatHistory([
      ...chatHistory,
      { role: 'assistant', content: 'Database connected successfully!' },
      { role: 'assistant', content: data.description },
    ]);
  };

  const handleNewMessage = (message) => {
    setChatHistory([...chatHistory, message]);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        minHeight: '100vh',
        backgroundColor: 'background.default',
        py: 4
      }}>
        <Container maxWidth="md">
          <Typography 
            variant="h4" 
            gutterBottom 
            align="center"
            sx={{ 
              mb: 4,
              fontWeight: 'bold',
              color: 'primary.main'
            }}
          >
            DBBuddy: Your Smart Database Assistant
          </Typography>
          
          {!isConnected && (
            <UploadSection onUpload={handleUpload} onConnect={handleConnect} />
          )}
          
          <ChatBox chatHistory={chatHistory} />
          
          {isConnected && (
            <QuerySection onNewMessage={handleNewMessage} />
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;