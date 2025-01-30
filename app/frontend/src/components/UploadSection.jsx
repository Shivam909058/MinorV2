import React, { useState } from 'react';
import { Box, Button, Typography, TextField } from '@mui/material';
import axios from 'axios';

const UploadSection = ({ onUpload, onConnect }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [connectionString, setConnectionString] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await axios.post('http://localhost:8000/upload-db/', formData);
      onUpload(response.data);
    } catch (error) {
      alert(error.response.data.detail);
    }
  };

  const handleConnect = async () => {
    if (!connectionString) return;
    const formData = new FormData();
    formData.append('connection_string', connectionString);
    try {
      const response = await axios.post('http://localhost:8000/connect-db/', formData);
      onConnect(response.data);
    } catch (error) {
      alert(error.response.data.detail);
    }
  };

  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Upload Database File
      </Typography>
      <input type="file" accept=".db, .sqlite, .postgresql, .mysql" onChange={handleFileChange} />
      <Button variant="contained" sx={{ ml: 2 }} onClick={handleUpload}>Upload</Button>

      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Or Connect via Connection String
        </Typography>
        <TextField
          fullWidth
          label="Connection String"
          value={connectionString}
          onChange={(e) => setConnectionString(e.target.value)}
        />
        <Button variant="contained" sx={{ mt: 2 }} onClick={handleConnect}>Connect</Button>
      </Box>
    </Box>
  );
};

export default UploadSection;