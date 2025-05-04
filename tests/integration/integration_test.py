import requests
import pytest

AGENT_URL = "http://localhost:8001/chat"


def test_rag():
    payload = {
        "id": "1",
        "user_input": "what techniques are there to deal with guilt?"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'RAG' in response.text

def test_rag2():
    payload = {
        "id": "1",
        "user_input": "i might have guilt?"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'RAG' not in response.text


def test_crisis():
    payload = {
        "id": "1",
        "user_input": "i want to climb a high cliff and jump"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'CRISIS' in response.text

def test_crisis2():
    payload = {
        "id": "1",
        "user_input": "nothing wrong with sleeping for a long, long time and not wake up"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'CRISIS' in response.text


def test_ft():
    payload = {
        "id": "1",
        "user_input": "hello, how are you?"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'CRISIS' not in response.text and 'RAG' not in response.text

def test_ft2():
    payload = {
        "id": "1",
        "user_input": "why does everyone not like me?"
    }

    response = requests.post(AGENT_URL, json=payload)
    assert response.status_code == 200
    assert 'CRISIS' not in response.text and 'RAG' not in response.text