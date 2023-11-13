from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance
from python_backend.contract.deploy import deploy_contract as deploy
import python_backend.contract.deploy
from web3.middleware import geth_poa_middleware
from transformers import TapasTokenizer
import scipy
import python_backend.controllers.tokens_bert as tokens
from urllib import parse
import python_backend.controllers.html_reader as reader
import operator

from flask import jsonify, request
import sys, os, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import tensorflow as tf
from PIL import Image
import cv2, io, librosa
from io import BytesIO
import openvino as ov

sys.dont_write_bytecode = True

def predictWound():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'wound', 'optimized_wound.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'wound', 'wound_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    prediction = "This wound is likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    return Success("Success", prediction, 200)

def predictMedicine(image_file):
    
    # image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'optimized_medicine.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        
        # image = Image.open(BytesIO(imagePath))
        image = imagePath.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'medicine_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    # prediction = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
    #         class_names[np.argmax(score)], 100 * np.max(score)
    #     )
    prediction = class_names[np.argmax(score)]
    return prediction
    # return Success("Success", prediction, 200)


def predictPrescription():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    detection_xml = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'intel', 'horizontal-text-detection-0001', 'FP16',
     'horizontal-text-detection-0001.xml')
    detection_bin = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'intel', 'horizontal-text-detection-0001', 'FP16','horizontal-text-detection-0001.bin')

    detection_model = core.read_model(
            model=detection_xml, weights=detection_bin
        )
    
    detection_compiled_model = core.compile_model(model=detection_model, device_name="CPU")
    detection_input_layer = detection_compiled_model.input(0)

    image = load_image(image_file)

    N, C, H, W = detection_input_layer.shape

    resized_image = cv2.resize(image, (W, H))

    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    output_key = detection_compiled_model.output("boxes")
    boxes = detection_compiled_model([input_image])[output_key]

    boxes = boxes[~np.all(boxes == 0, axis=1)]
    # /mnt/hdd/projects/codeshift/python_backend/trained_models/charater_recognition/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.xml
    recognition_xml = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'public', 'text-recognition-resnet-fc', 'FP16',
     'text-recognition-resnet-fc.xml')
    recognition_bin = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'public', 'text-recognition-resnet-fc', 'FP16','text-recognition-resnet-fc.bin')

    recognition_model = core.read_model(
        model=recognition_xml, weights=recognition_bin
    )

    recognition_compiled_model = core.compile_model(model=recognition_model, device_name="CPU")

    recognition_output_layer = recognition_compiled_model.output(0)
    recognition_input_layer = recognition_compiled_model.input(0)

    # Get the height and width of the input layer.
    _, _, H, W = recognition_input_layer.shape

    # Calculate scale for image resizing.
    (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the image to grayscale for the text recognition model.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get a dictionary to encode output, based on the model documentation.
    letters = "~0123456789abcdefghijklmnopqrstuvwxyz"

    # Prepare an empty list for annotations.
    annotations = list()
    cropped_images = list()
    # fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
    # Get annotations for each crop, based on boxes given by the detection model.
    for i, crop in enumerate(boxes):
        # Get coordinates on corners of a crop.
        (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
        image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))

        # Run inference with the recognition model.
        result = recognition_compiled_model([image_crop])[recognition_output_layer]

        # Squeeze the output to remove unnecessary dimension.
        recognition_results_test = np.squeeze(result)

        # Read an annotation based on probabilities from the output layer.
        annotation = list()
        for letter in recognition_results_test:
            parsed_letter = letters[letter.argmax()]

            # Returning 0 index from `argmax` signalizes an end of a string.
            if parsed_letter == letters[0]:
                break
            annotation.append(parsed_letter)
        annotations.append("".join(annotation))
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        cropped_images.append(cropped_image)

    boxes_with_annotations = list(zip(boxes, annotations))
    allPrescription = []
    for cropped_image, annotation in zip(cropped_images, annotations):
        allPrescription.append(predictMedicine(cropped_image))
        print(annotation)
    return Success("Success", allPrescription, 200)



def load_image(image_file) -> np.ndarray:
    
    
    # if path.startswith("http"):
    #     # Set User-Agent to Mozilla because some websites block
    #     # requests with User-Agent Python
    #     response = requests.get(path, headers={"User-Agent": "Mozilla/5.0"})
    #     array = np.asarray(bytearray(response.content), dtype="uint8")
    #     image = cv2.imdecode(array, -1)  # Loads the image as BGR
    # else:
        # image = cv2.imread(image)
    image_data = image_file.read()
        
    # Process the image (e.g., convert to OpenCV format)
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


def multiply_by_ratio(ratio_x, ratio_y, box):
    return [
        max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
        for idx, shape in enumerate(box[:-1])
    ]


def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img


def questionAnswering():
    question = list(request.json.values())
    # sources = ["A variety of medications serve diverse purposes in addressing health concerns. Aspirin, functioning as both a pain reliever and a blood thinner, is commonly employed to alleviate pain and inflammation. Ibuprofen, a nonsteroidal anti-inflammatory drug (NSAID), is utilized for pain relief, inflammation reduction, and fever reduction. Acetaminophen is favored for mild to moderate pain relief and fever reduction. Loratadine and cetirizine are antihistamines employed to manage allergy symptoms such as sneezing and itching. Diphenhydramine, another antihistamine, serves dual purposes by alleviating allergies and acting as a sleep aid. Medications like ranitidine, omeprazole, and lansoprazole are used to reduce stomach acid, treating conditions like heartburn and ulcers. Statins such as simvastatin and atorvastatin aim to lower cholesterol levels, decreasing the risk of heart disease. Metformin controls blood sugar levels in type 2 diabetes. Medications like levothyroxine address hypothyroidism by providing thyroid hormone replacement. Prednisone, a corticosteroid, suppresses inflammation and the immune system. Albuterol, a bronchodilator, is employed for asthma and chronic obstructive pulmonary disease (COPD). Sertraline, lisinopril, and amlodipine are medications for depression, hypertension, and high blood pressure, respectively. Anticoagulants like warfarin and antiplatelets like clopidogrel prevent blood clot formation. Hydrochlorothiazide and furosemide are diuretics used for hypertension and fluid retention. Duloxetine, venlafaxine, escitalopram, and citalopram are medications addressing depression and anxiety disorders. Aripiprazole, quetiapine, and olanzapine are atypical antipsychotics for schizophrenia and bipolar disorder. Benzodiazepines like alprazolam and lorazepam are used to manage anxiety and panic disorders. Zolpidem is a sedative-hypnotic for insomnia. Stimulants like methylphenidate treat attention deficit hyperactivity disorder (ADHD). Inhaled corticosteroids such as fluticasone and leukotriene receptor antagonists like montelukast address asthma and allergies. Antidiabetic medications like sitagliptin are used for type 2 diabetes. Mirtazapine, gabapentin, and pregabalin serve purposes ranging from managing depression to treating neuropathic pain. Opioid analgesics like hydrocodone/acetaminophen, oxycodone/acetaminophen, morphine, and tramadol are employed for varying degrees of pain relief. Dextromethorphan is a cough suppressant, guaifenesin is an expectorant, and miconazole is an antifungal medication for coughs, congestion, and fungal infections, respectively. Each medication addresses specific health needs, and their usage should align with professional guidance for optimal effectiveness and safety."]

    sources = python_backend.contract.deploy.medicine_table

    core = ov.Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_xml = os.path.join(current_directory, '..', 'trained_models', 'question_answering', 'bert-small-uncased-whole-word-masking-squad-int8-0002.xml')
    model = core.read_model(model_xml)

    compiled_model = core.compile_model(model=model, device_name="CPU")

    input_keys = list(compiled_model.inputs)
    output_keys = list(compiled_model.outputs)

    # Get the network input size.
    input_size = compiled_model.input(0).shape[1]

    # Download the vocabulary from the openvino_notebooks storage
    vocab_file_path = os.path.join(current_directory, '..', 'trained_models', 'question_answering', 'vocab.txt')

    # Create a dictionary with words and their indices.
    vocab = tokens.load_vocab_file(str(vocab_file_path))

    # Define special tokens.
    cls_token = vocab["[CLS]"]
    pad_token = vocab["[PAD]"]
    sep_token = vocab["[SEP]"]



    def load_context(sources):
        input_urls = []
        paragraphs = []
        for source in sources:
            result = parse.urlparse(source)
            if all([result.scheme, result.netloc]):
                input_urls.append(source)
            else:
                paragraphs.append(source)

        paragraphs.extend(reader.get_paragraphs(input_urls))
    
        return "\n".join(paragraphs)
    def prepare_input(question_tokens, context_tokens):
        # A length of question in tokens.
        question_len = len(question_tokens)
        # The context part size.
        context_len = input_size - question_len - 3

        if context_len < 16:
            raise RuntimeError("Question is too long in comparison to input size. No space for context")

        # Take parts of the context with overlapping by 0.5.
        for start in range(0, max(1, len(context_tokens) - context_len), context_len // 2):
            # A part of the context.
            part_context_tokens = context_tokens[start:start + context_len]
            # The input: a question and the context separated by special tokens.
            input_ids = [cls_token] + question_tokens + [sep_token] + part_context_tokens + [sep_token]
            # 1 for any index if there is no padding token, 0 otherwise.
            attention_mask = [1] * len(input_ids)
            # 0 for question tokens, 1 for context part.
            token_type_ids = [0] * (question_len + 2) + [1] * (len(part_context_tokens) + 1)

            # Add padding at the end.
            (input_ids, attention_mask, token_type_ids), pad_number = pad(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        token_type_ids=token_type_ids)

            # Create an input to feed the model.
            input_dict = {
                "input_ids": np.array([input_ids], dtype=np.int32),
                "attention_mask": np.array([attention_mask], dtype=np.int32),
                "token_type_ids": np.array([token_type_ids], dtype=np.int32),
            }

            # Some models require additional position_ids.
            if "position_ids" in [i_key.any_name for i_key in input_keys]:
                position_ids = np.arange(len(input_ids))
                input_dict["position_ids"] = np.array([position_ids], dtype=np.int32)

            yield input_dict, pad_number, start


    # A function to add padding.
    def pad(input_ids, attention_mask, token_type_ids):
        # How many padding tokens.
        diff_input_size = input_size - len(input_ids)

        if diff_input_size > 0:
            # Add padding to all the inputs.
            input_ids = input_ids + [pad_token] * diff_input_size
            attention_mask = attention_mask + [0] * diff_input_size
            token_type_ids = token_type_ids + [0] * diff_input_size

        return (input_ids, attention_mask, token_type_ids), diff_input_size
    
    def postprocess(output_start, output_end, question_tokens, context_tokens_start_end, padding, start_idx):

        def get_score(logits):
            out = np.exp(logits)
            return out / out.sum(axis=-1)

        # Get start-end scores for the context.
        score_start = get_score(output_start)
        score_end = get_score(output_end)

        # An index of the first context token in a tensor.
        context_start_idx = len(question_tokens) + 2
        # An index of the last+1 context token in a tensor.
        context_end_idx = input_size - padding - 1

        # Find product of all start-end combinations to find the best one.
        max_score, max_start, max_end = find_best_answer_window(start_score=score_start,
                                                                end_score=score_end,
                                                                context_start_idx=context_start_idx,
                                                                context_end_idx=context_end_idx)

        # Convert to context text start-end index.
        max_start = context_tokens_start_end[max_start + start_idx][0]
        max_end = context_tokens_start_end[max_end + start_idx][1]

        return max_score, max_start, max_end

    def find_best_answer_window(start_score, end_score, context_start_idx, context_end_idx):
        context_len = context_end_idx - context_start_idx
        score_mat = np.matmul(
            start_score[context_start_idx:context_end_idx].reshape((context_len, 1)),
            end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
        )
        # Reset candidates with end before start.
        score_mat = np.triu(score_mat)
        # Reset long candidates (>16 words).
        score_mat = np.tril(score_mat, 16)
        # Find the best start-end pair.
        max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
        max_score = score_mat[max_s, max_e]

        return max_score, max_s, max_e


    def get_best_answer(question, context):
        # Convert the context string to tokens.
        context_tokens, context_tokens_start_end = tokens.text_to_tokens(text=context.lower(),
                                                                        vocab=vocab)
        # Convert the question string to tokens.
        question_tokens, _ = tokens.text_to_tokens(text=question.lower(), vocab=vocab)

        results = []
        # Iterate through different parts of the context.
        for network_input, padding, start_idx in prepare_input(question_tokens=question_tokens,
                                                            context_tokens=context_tokens):
            # Get output layers.
            output_start_key = compiled_model.output("output_s")
            output_end_key = compiled_model.output("output_e")

            # OpenVINO inference.
            result = compiled_model(network_input)
            # Postprocess the result, getting the score and context range for the answer.
            score_start_end = postprocess(output_start=result[output_start_key][0],
                                        output_end=result[output_end_key][0],
                                        question_tokens=question_tokens,
                                        context_tokens_start_end=context_tokens_start_end,
                                        padding=padding,
                                        start_idx=start_idx)
            results.append(score_start_end)

        # Find the highest score.
        answer = max(results, key=operator.itemgetter(0))
        # Return the part of the context, which is already an answer.
        return context[answer[1]:answer[2]], answer[0]
    
    def run_question_answering(sources, example_question=None):
        context = load_context(sources)

        if len(context) == 0:
            print("Error: Empty context or outside paragraphs")
            return

        if example_question is not None:
            answer, score = get_best_answer(question=example_question, context=context)
           
            return answer
            
        else:
            while True:
                question = input()
                # if no question - break
                if question == "":
                    break

                # measure processing time
                answer, score = get_best_answer(question=question, context=context)
                return answer

    answer = run_question_answering(sources, example_question=question[0])
    return Success("Success", answer, 200)
def predictIndividualMedicine():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'optimized_medicine.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'medicine_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    prediction = "The prescription submitted  belongs to {} medicine".format(
            class_names[np.argmax(score)]
        )
    return Success("Success", prediction, 200)


def speechToText():
    # print(request.files)
    # if 'speech' not in request.files:
    #     return jsonify({'error': 'No file part'})
    alphabet = " abcdefghijklmnopqrstuvwxyz'~"
    # file_name = request.files['speech']
    file_name = "/home/mktetts/Documents/codeshift/models/speech_to_text/data/recorded-audio.wav"
    audio, sampling_rate = librosa.load(path=str(file_name), sr=16000)
    print("1")
    if max(np.abs(audio)) <= 1:
        audio = (audio * (2**15 - 1))
    audio = audio.astype(np.int16)
    def audio_to_mel(audio, sampling_rate):
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

        # Calculate the window length.
        win_length = round(sampling_rate * 0.02)

        # Based on the previously calculated window length, run short-time Fourier transform.
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
                    win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))

        # Create mel filter-bank, produce transformation matrix to project current values onto Mel-frequency bins.
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0, htk=False)
        return mel_basis, spec
    print("1")


    def mel_to_input(mel_basis, spec, padding=16):
        # Convert to a logarithmic scale.
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

        # Normalize the output.
        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)

        # Calculate padding.
        remainder = normalized.shape[1] % padding
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, padding - remainder)))[None]
        return normalized[None]
    
    mel_basis, spec = audio_to_mel(audio=audio.flatten(), sampling_rate=sampling_rate)
    print("1")

    audio = mel_to_input(mel_basis=mel_basis, spec=spec)
    core = ov.Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_xml = os.path.join(current_directory, '..', 'trained_models', 'speech_to_text', 'speech_to_text.xml')
    model = core.read_model(
        model=model_xml
        )
    model_input_layer = model.input(0)
    shape = model_input_layer.partial_shape
    shape[2] = -1
    model.reshape({model_input_layer: shape})
    compiled_model = core.compile_model(model=model, device_name="CPU")
    character_probabilities = compiled_model([ov.Tensor(audio)])[0]
    # Remove unnececery dimension
    character_probabilities = np.squeeze(character_probabilities)
    print("1")

    # Run argmax to pick most possible symbols
    character_probabilities = np.argmax(character_probabilities, axis=1)
    def ctc_greedy_decode(predictions):
        previous_letter_id = blank_id = len(alphabet) - 1
        transcription = list()
        for letter_index in predictions:
            if previous_letter_id != letter_index != blank_id:
                transcription.append(alphabet[letter_index])
            previous_letter_id = letter_index
        return ''.join(transcription)
    transcription = ctc_greedy_decode(character_probabilities)
    print("trasdcc", transcription)
    return Success("Success", transcription, 200)