from utils import get_embedding
from flask import jsonify
from config import *
from flask import current_app

import openai

from config import *

TOP_K = 20
#global_counter = 0
openai.api_key="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"
global_initial_prompt = f"Given a question, try to answer it using the content of the work order file extracts supplied with the question, and if you cannot answer, or find " \
                        f"a relevant file, just output \"I couldn't find the answer to that question in your files.\".\n\n" \
                        f"The files uploaded are work orders, and you may find information related to Service Date, Delivery Date, and Due Date, which refer to the " \
                        f"same primary date field for the work orders. Combine anything under Service Details, Services to be Performed, Instructions, Origin Items, and " \
                        f"Destination Items into the primary description field for the work order for use in your responses. You will always be thorough in your response, giving as much info on jobs I ask about as possible. If I ask about details for a job, you could respond by giving me as many fields from that job you can extract." \
                        f"Always use the previous context of the conversation when responding to a question. If the question is not actually a question, respond with \"That's not a valid question.\"\n\n" \
                        f"If the answer is not contained in the files or if there are no file extracts, respond with \"I couldn't find the answer " \
                        f"to that question in your files.\" If the question is not actually a question, respond with \"That's not a valid question.\"\n\n" \
                        f"In the cases where you can find the answer, first give the answer. Then explain how you found the answer from the source or sources, " \
                        f"and use the exact filenames of the source files you mention. Do not make up the names of any other files other than those mentioned "\
                        f"in the files context. Give the answer in markdown format." \
                        f"Use the following format:\n\nQuestion: <question>\n\nFiles:\n<###\n\"filename 1\"\nfile text>\n<###\n\"filename 2\"\nfile text>...\n\n"\
                        f"Answer: <answer or \"I couldn't find the answer to that question in your files\" or \"That's not a valid question.\">\n\n" \
                        
messages = [{
                        "role": "system",
                        "content": global_initial_prompt
                        #f"Example Below:" \
                        #f"Question: What is my name?\n\n" \
                        #f"Files:\n My name is Tom and I have a horse\n" \
                        #f"Answer: "
                        #f"I am going to give you the content of Files one time, and you will have to answer the questions based on the content of the files. " \
                    },]

def rewrite_question(question,msgs):
    logging.info(f"Rewriting question: {question}")
    
    messages.append({"role": "assistant", "content":f"You will now use the context of the conversation to help me rewrite my question to be as clear as possible.  The new prompt should contain as many fields that would allow a database search on them as possible." })
    messages.append({"role": "user", "content":f"Rewrite the following question to include as much context as possible for a search in a database: '{question}'"})
    response = openai.ChatCompletion.create(
            messages=messages,
            model=GENERATIVE_MODEL,
            max_tokens=1000,
            temperature=0.7,
        )
    choices = response["choices"]  # type: ignore
    answer = choices[0].message.content.strip()
    messages.append({"role": "system", "content":f"{answer}"})
    messages.append({"role": "assistant", "content":f"{global_initial_prompt}"})

    return answer
 

def get_answer_from_files(question, session_id, pinecone_index):
    logging.info(f"Getting answer for question: {question}")

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)

    try:
        query_response = pinecone_index.query(
            namespace=session_id,
            top_k=TOP_K,
            include_values=True,
            include_metadata=True,
            vector=search_query_embedding,
        )
        logging.info(
            f"[get_answer_from_files] received query response from Pinecone: {query_response}")

        files_string = ""
        file_text_dict = current_app.config["file_text_dict"]

        for i in range(len(query_response.matches)):
            result = query_response.matches[i]
            file_chunk_id = result.id
            score = result.score
            filename = result.metadata["filename"]
            file_text = file_text_dict.get(file_chunk_id)
            file_string = f"###\n\"{filename}\"\n{file_text}\n"
            if score < COSINE_SIM_THRESHOLD and i > 0:
                logging.info(
                    f"[get_answer_from_files] score {score} is below threshold {COSINE_SIM_THRESHOLD} and i is {i}, breaking")
                break
            files_string += file_string
        
        # Note: this is not the proper way to use the ChatGPT conversational format, but it works for now
        
       
        
        #messages.append(
        #    {
        #        "role": "system",
        #        "content": f"Given a question, try to answer it using the content of the work order file extracts below. The files uploaded are work orders, and you may find information related to Service Date, Delivery Date, and Due Date, which refer to the same primary date field for the work orders. Combine anything under Service Details, Services to be Performed, Instructions, Origin Items, and Destination Items into the primary description field for the work order for use in your responses. If you cannot find the answer, or if the question is not clear, guide the user on how to improve their prompt or question to get their intended result.\n\n" \
        #        f"Use the following format:\n\nQuestion: <question>\n\nFiles:\n<###\n\"filename 1\"\nfile text>\n<###\n\"filename 2\"\nfile text>...\n\n" \
        #        f"Answer: <answer or guidance on how to improve the question>\n\n" \
        #    },
        #    {
        #        "role": "system",
        #        "content": f"Question: {question}\n\n" \
        #        f"Files:\n{files_string}\n" \
        #        f"Answer:"
        #    }
        #)
        #if global_counter > 0:
        #    rewrittenQuestion = rewrite_question(question,messages)
        #    messages.append(
        #    {   "role": "user",
        #        "content":
        #        f"Question: {rewrittenQuestion}\n\n" \
        #        f"Files:\n{files_string}\n" \
        #        f"Answer:"
        #    },
        #    )
        #else:
        #    messages.append(
        #    {   "role": "user",
        #        "content":
        #        f"Question: {question}\n\n" \
        #        f"Files:\n{files_string}\n" \
        #        f"Answer:"
        #    },
        #    )
        if len(messages) >1:
            question = rewrite_question(question,messages)
        messages.append(
            {   "role": "user",
                "content":
                f"Question: {question}\n\n" \
                f"Files:\n{files_string}\n" \
                f"Answer:"
            },
            )
        response = openai.ChatCompletion.create(
            messages=messages,
            model=GENERATIVE_MODEL,
            max_tokens=1000,
            temperature=0,
        )

        choices = response["choices"]  # type: ignore
        answer = choices[0].message.content.strip()
        messages.append(
            {
                "role": "system",
                "content": f"{answer}\n\n"
            }
        )
        messagesStr = ''.join(str(e) for e in messages)
        #messagesStr = messagesStr.replace("'}{'", "'}<br /><br />{'")


        #answer.append(messagesStr)

        logging.info(f"[get_answer_from_files] answer: {answer}")
        #global_counter = global_counter + 1
        return jsonify({"answer": answer, "messagesStr": messagesStr})

    except Exception as e:
        logging.info(f"[get_answer_from_files] error: {e}")
        return str(e)
