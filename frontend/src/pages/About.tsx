import React from 'react';

const About: React.FC = () => {
  return (
    <div className="about">
      <div className="about__content">
        <h2>About AI Image Caption Generator</h2>
        
        <section className="about__section">
          <h3>How It Works</h3>
          <p>
            Our AI Image Caption Generator uses a sophisticated deep learning architecture
            that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM)
            networks to analyze images and generate natural language descriptions.
          </p>
        </section>

        <section className="about__section">
          <h3>Technology Stack</h3>
          <ul>
            <li><strong>Frontend:</strong> React with TypeScript</li>
            <li><strong>Backend:</strong> FastAPI with Python</li>
            <li><strong>Machine Learning:</strong> TensorFlow</li>
            <li><strong>Deployment:</strong> Google Cloud Platform</li>
          </ul>
        </section>

        <section className="about__section">
          <h3>Features</h3>
          <ul>
            <li>Real-time image caption generation</li>
            <li>Support for multiple image formats</li>
            <li>Responsive web interface</li>
            <li>Fast processing with cloud deployment</li>
            <li>Error handling and user feedback</li>
          </ul>
        </section>
      </div>
    </div>
  );
};

export default About;