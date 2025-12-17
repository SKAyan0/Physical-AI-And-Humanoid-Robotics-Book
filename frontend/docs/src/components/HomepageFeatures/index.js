import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Embodied Intelligence',
    Svg: require('@site/static/img/embodied-intelligence.svg').default,
    description: (
      <>
        Learn how to bridge the gap between digital AI and physical robotics.
        Understand how abstract algorithms translate to embodied behaviors in physical systems.
      </>
    ),
  },
  {
    title: 'Hands-on Learning',
    Svg: require('@site/static/img/hands-on-learning.svg').default,
    description: (
      <>
        Practical implementation of ROS 2, NVIDIA Isaac, and VLA (Vision-Language-Action) models
        with interactive examples and code snippets.
      </>
    ),
  },
  {
    title: 'Interactive AI Tutor',
    Svg: require('@site/static/img/interactive-ai-tutor.svg').default,
    description: (
      <>
        Get instant answers to your questions about robotics concepts with our
        integrated RAG chatbot that understands the book content.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}