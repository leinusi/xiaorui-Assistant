# Xiaorui-Assistant

Xiaorui-Assistant is a streaming large-scale model assistant web app developed using the Django framework. It leverages streaming technology to enable real-time interaction with large language models.

## Features

- Utilizes advanced large-scale language models to provide intelligent conversation and question-answering services
- Employs streaming technology to enable real-time responses from the assistant
- Uses Django as the web framework, providing stable and efficient backend support
- Offers a user-friendly interface for seamless interaction between users and the assistant
- Supports concurrent access by multiple users, capable of handling high-concurrency requests

## Installation and Deployment

1. Clone the project code to your local machine:
```git clone https://github.com/yourusername/xiaorui-assistant.git```
2. Navigate to the project directory:
```cd xiaorui-assistant```
3. Create and activate a virtual environment (optional but recommended):
```python -m venv venv```
```source venv/bin/activate```
4. Install project dependencies:
```pip install -r requirements.txt```
5. Run database migrations:
```python manage.py migrate```
6. Start the Django development server:
```python manage.py runserver```
7. Access the Xiaorui-Assistant application in your web browser at `http://localhost:6006`.

## Configuration

- Application configuration, such as database settings and language model API parameters, can be modified in the `xiaorui_assistant/settings.py` file.
- The language model API configuration requires providing the appropriate API key and endpoint URL. Please refer to the relevant documentation for more details.

## Contribution Guidelines

Contributions to the Xiaorui-Assistant project are welcome! If you encounter any issues or have suggestions for improvement, please submit an issue or pull request.

Before submitting a pull request, ensure that your code adheres to the project's coding standards and passes all test cases.

## License

The Xiaorui-Assistant project is released under the [MIT License](LICENSE).


Thank you for your interest and support for the Xiaorui-Assistant project!
