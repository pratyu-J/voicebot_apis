from flask import Flask, request, jsonify, g, make_response, send_file
from flask_cors import CORS
# from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy.orm import Session
import openai 
from pathlib import Path
import os
from decouple import config


# Import session and model setup
# from model import SessionLocal, Conversation
from gen_faiss_index import get_index_directory, get_response, logger, getVoiceResponse

app = Flask(__name__)
CORS(app)

app.json.ensure_ascii = False


openai_key = config("OPENAI_API_KEY")  # Open AI API token
openai.api_key = openai_key

# def get_db():
#     if 'db' not in g:
#         g.db = SessionLocal()
#     return g.db

@app.route('/api/questions', methods=['GET', 'POST'])
def receiveQuestions():
    try:
        value = request.headers.get('authorization')
        user_id = value.split()[1] if value else None
        question = request.args.get('question')
        servicetype = request.headers.get('servicetype')
        usertype = request.headers.get('usertype')
        
        
        response = get_index_directory(user_id ,servicetype, usertype,question)
        # response = get_response(chain, question)
        
        
        # db_session: Session = get_db()
        # try:
        #     conversation = Conversation(
        #         sender=user_id,
        #         message=question,
        #         response=str(response),
        #     )
        #     db_session.add(conversation)
        #     db_session.commit()
        #     logger.info("Conversation #%s stored in database", conversation.id)
        # except SQLAlchemyError as err:
        #     db_session.rollback()
        #     logger.error("Error storing conversation in database: %s", err)
        return jsonify({"ans": response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/voiced', methods=['GET', 'POST'])
def receiveVoiceQuestions():
    try:
        print("hello")
        value = request.headers.get('authorization')
        print("aith=", value)
        user_id = value.split()[1] if value else None
        question = request.args.get('question')
        
        
        
        response = getVoiceResponse(question, user_id)
        # response = get_response(chain, question)
        
        
        # db_session: Session = get_db()
        # try:
        #     conversation = Conversation(
        #         sender=user_id,
        #         message=question,
        #         response=str(response),
        #     )
        #     db_session.add(conversation)
        #     db_session.commit()
        #     logger.info("Conversation #%s stored in database", conversation.id)
        # except SQLAlchemyError as err:
        #     db_session.rollback()
        #     logger.error("Error storing conversation in database: %s", err)
        return jsonify({"ans": response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/speech', methods=['GET', 'POST'])
def genAudio():
    print("getting speech")
    text = request.args.get('text')
    
    try:
        data = request.get_json()
        text = data.get("input")

        if( not text):
            return jsonify({'error': 'No text provided'}), 400
        print("near file")
        speech_file_path = Path(__file__).parent / "speech.pm3"


        response = openai.audio.speech.create(model="tts-1", input = text, voice='alloy')
        print("converted to audio ")

        response.stream_to_file(speech_file_path)

        return send_file(
            speech_file_path,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name = "speech.mp3"
        )
        # audio_data = response.content  # Extract audio content

        # # Create Flask response
        # flask_response = make_response(audio_data)
        # flask_response.headers['Content-Type'] = 'audio/webm'  # Set content type (adjust if needed)
        # return flask_response, 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
