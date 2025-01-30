import React, { useState } from 'react';
import { Box, TextField, Button } from '@mui/material';
import axios from 'axios';

const QuerySection = ({ onNewMessage }) => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleQuery = async () => {
    if (!query) return;
    setIsLoading(true);
    onNewMessage({ role: 'user', content: query });

    try {
      const response = await axios.post('http://localhost:8000/query/', { query });
      
      if (response.data.success) {
        // Show the SQL query if available
        if (response.data.query) {
          onNewMessage({ 
            role: 'assistant', 
            content: `Generated SQL Query:\n\`\`\`sql\n${response.data.query}\n\`\`\`` 
          });
        }
        
        // Show the results if available
        if (response.data.result) {
          onNewMessage({ 
            role: 'assistant', 
            content: response.data.result 
          });
        }
        
        // Show the explanation
        if (response.data.explanation) {
          onNewMessage({ 
            role: 'assistant', 
            content: response.data.explanation 
          });
        }
      } else {
        onNewMessage({ 
          role: 'assistant', 
          content: response.data.explanation || 'Sorry, I could not process your query.' 
        });
      }
    } catch (error) {
      onNewMessage({ 
        role: 'assistant', 
        content: error.response?.data?.detail || 'An error occurred while processing your query.'
      });
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  };

  return (
    <Box sx={{ display: 'flex', mt: 2 }}>
      <TextField
        fullWidth
        multiline
        rows={2}
        label="Ask me anything about your database..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyPress={(e) => { 
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
          }
        }}
        disabled={isLoading}
      />
      <Button 
        variant="contained" 
        sx={{ ml: 2, minWidth: '100px' }} 
        onClick={handleQuery}
        disabled={isLoading}
      >
        {isLoading ? 'Processing...' : 'Send'}
      </Button>
    </Box>
  );
};

export default QuerySection; 