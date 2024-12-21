require('dotenv').config();

const express = require('express');
const axios = require('axios');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const upload = multer(); // For handling form-data
const openai = require('openai');


openai.apiKey = process.env.OPENAI_API_KEY
const app = express();
app.use(cors());
app.use(express.json());

const WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions";

app.post('/transcribe', upload.single('file'), async (req, res) => {
    console.log('Received request at /transcribe');
  try {

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }
      
      const formData = new FormData();  
      console.log(req.file.buffer);
      formData.append('file', req.file.buffer, {
        filename: 'audio.webm', // Use a suitable name
        contentType: 'audio/webm' // Adjust if using a different audio format
      });
      console.log('Received request at /transcribe2');
      formData.append('model', 'whisper-1'); // Specify the Whisper model as required
      console.log("set up whisper api")
    const response = await axios.post(WHISPER_API_URL, formData, {
      headers: {
        'Authorization': `Bearer ${openai.apiKey}`,
        ...formData.getHeaders()
      }
    });
    console.log("sent to whisper api");
    console.log("response", response)
    res.json(response.data.text);
  } catch (error) {
    res.status(error.response?.status || 500).json(error.response?.data || { error: 'Server error' });
  }
});


// app.post('/speech', async (req, res) => {
//   const text = req.body.text;
//   const response = await openai.Audio.speech.create({
//     model: 'tts-1',
//     input: text,
//     voice: 'female',   

//   });
//   res.setHeader('Content-Type, response.contentType');
//   res.send(response.data);
// });

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});