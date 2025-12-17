import React, { useState, useEffect, useRef } from 'react';
import './RagChatWidget.css';

const RagChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [selectionPosition, setSelectionPosition] = useState({ x: 0, y: 0 });
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  // Debug log to verify component is loaded
  useEffect(() => {
    console.log('RagChatWidget component loaded');
  }, []);

  // Function to get selected text from the page and track its position
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        // Get the position of the selection
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          setSelectionPosition({
            x: rect.left, // Position at the left of selection
            y: rect.top - 35    // Position just above the selection
          });
        }
        setSelectedText(selectedText);
      } else {
        setSelectedText('');
        // Clear the position when no text is selected
        setSelectionPosition({ x: 0, y: 0 });
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Create new session
  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await fetch('https://beastfrog145-h1chat.hf.space/api/v1/sessions/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            initial_context: 'Physical AI & Humanoid Robotics Book Assistant'
          })
        });

        if (response.ok) {
          const data = await response.json();
          setSessionId(data.session_id);
        } else {
          console.error('Failed to create session:', await response.text());
        }
      } catch (error) {
        console.error('Error creating session:', error);
        // Add an error message to the chat if session creation fails
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: 'Warning: Unable to create chat session. Backend may be unavailable.',
          sender: 'bot',
          timestamp: new Date().toISOString()
        }]);
      }
    };

    if (!sessionId) {
      createSession();
    }
  }, [sessionId]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Prepare the request
      const requestBody = {
        message: inputMessage,
        session_id: sessionId,
        context: selectedText || undefined,
        metadata: {
          temperature: 0.7
        }
      };

      // Send message to backend
      const response = await fetch('https://beastfrog145-h1chat.hf.space/api/v1/chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          id: Date.now() + 1,
          text: data.response,
          sender: 'bot',
          sources: data.sources || [],
          timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, botMessage]);
        setSelectedText(''); // Clear selected text after use
      } else {
        let errorMessageText = 'Sorry, I encountered an error. Please try again.';

        // Handle specific error cases
        if (data.detail && typeof data.detail === 'string') {
          if (data.detail.includes('insufficient_quota')) {
            errorMessageText = 'API quota exceeded. Please check your OpenAI billing details.';
          } else if (data.detail.includes('401') || data.detail.includes('invalid_api_key')) {
            errorMessageText = 'Invalid API key. Please check your backend configuration.';
          } else {
            errorMessageText = data.detail;
          }
        } else if (data.detail && data.detail.error && data.detail.error.message) {
          errorMessageText = data.detail.error.message;
        }

        const errorMessage = {
          id: Date.now() + 1,
          text: errorMessageText,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      let errorMessageText = 'Sorry, I encountered an error. Please check that the backend is running.';

      // Check if it's a network error
      if (error.message.includes('fetch') || error.message.includes('network')) {
        errorMessageText = 'Cannot connect to backend. Please ensure the backend server is running on http://localhost:8000';
      }

      const errorMessage = {
        id: Date.now() + 1,
        text: errorMessageText,
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const askAboutSelection = async () => {
    if (!selectedText.trim()) {
      console.log('No selected text to send');
      return;
    }

    console.log('Ask AI button clicked, selected text:', selectedText);

    // Send the selected text directly to the AI first
    setInputMessage(selectedText);
    console.log('Input message set to:', selectedText);

    // Immediately call sendMessage instead of using setTimeout
    try {
      await sendMessage();
      console.log('Message sent successfully');

      // Then make sure the chat window is open to show the response
      if (!isOpen) {
        setIsOpen(true);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const closeAskButton = () => {
    setSelectedText('');
    setSelectionPosition({ x: 0, y: 0 });
  };

  return (
    <div className="rag-chat-widget">
      {/* Chat bubble/floating button */}
      <button
        className={`chat-toggle ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? '✕' : '🤖'}
      </button>

      {/* Floating Ask AI button near selected text */}
      {selectedText && (
        <div
          className="floating-ask-ai-container"
          style={{
            position: 'fixed',
            left: `${selectionPosition.x}px`,
            top: `${selectionPosition.y}px`,
            zIndex: 10000,
          }}
        >
          <button className="ask-ai-button" onClick={askAboutSelection}>Ask AI</button>
          <button className="close-ask-button" onClick={(e) => {
            e.stopPropagation(); // Prevent triggering the ask AI function
            closeAskButton();
          }}>×</button>
        </div>
      )}

      {/* Chat window */}
      {isOpen && (
        <div className="chat-window">
          <div className="chat-header">
            <h3>Robotics Book Assistant</h3>
            <p className="header-subtitle">Ask questions about the content</p>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your Robotics Book Assistant.</p>
                <p>You can ask me questions about the Physical AI & Humanoid Robotics content.</p>
                <p>Select text on the page and I can answer questions about it specifically!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.sender}`}
                >
                  <div className="message-content">
                    <p>{message.text}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="sources">
                        <small>Sources: {message.sources.join(', ')}</small>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={selectedText ? "Ask about selected text..." : "Ask about the robotics content..."}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="send-button"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RagChatWidget;