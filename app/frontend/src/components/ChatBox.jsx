import React, { useRef, useEffect } from 'react';
import { Box, Typography, Paper, Fade } from '@mui/material';
import { styled } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const MessageContainer = styled(Paper)(({ theme, role }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxWidth: '85%',
  borderRadius: 12,
  ...(role === 'user' ? {
    marginLeft: 'auto',
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
  } : {
    marginRight: 'auto',
    backgroundColor: theme.palette.grey[100],
  }),
}));

const ChatBox = ({ chatHistory }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const renderContent = (content) => {
    if (typeof content !== 'string') return content;

    return (
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={atomDark}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          table({ children }) {
            return (
              <Box sx={{ overflowX: 'auto', my: 2 }}>
                <table style={{ 
                  borderCollapse: 'collapse', 
                  width: '100%',
                  border: '1px solid #ddd'
                }}>
                  {children}
                </table>
              </Box>
            );
          },
          th({ children }) {
            return (
              <th style={{ 
                padding: '12px',
                backgroundColor: '#f5f5f5',
                borderBottom: '2px solid #ddd',
                textAlign: 'left'
              }}>
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td style={{ 
                padding: '8px',
                borderBottom: '1px solid #ddd'
              }}>
                {children}
              </td>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <Box sx={{ 
      height: '60vh',
      overflowY: 'auto',
      mb: 2,
      p: 2,
      backgroundColor: 'background.default',
      borderRadius: 2,
      boxShadow: 1
    }}>
      {chatHistory.map((chat, index) => (
        <Fade in={true} key={index} timeout={500}>
          <MessageContainer role={chat.role} elevation={1}>
            <Typography 
              variant="subtitle2" 
              color={chat.role === 'user' ? 'inherit' : 'text.secondary'}
              sx={{ mb: 1, fontWeight: 'bold' }}
            >
              {chat.role === 'user' ? 'You' : 'DBBuddy'}
            </Typography>
            {renderContent(chat.content)}
          </MessageContainer>
        </Fade>
      ))}
      <div ref={messagesEndRef} />
    </Box>
  );
};

export default ChatBox;