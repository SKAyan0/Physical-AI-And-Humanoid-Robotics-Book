/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/intro',
        'module1/chapter1.1',
        'module1/chapter1.2',
        'module1/chapter1.3',
      ],
      link: {
        type: 'doc',
        id: 'module1/intro',
      },
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/intro',
        'module2/chapter2.1',
        'module2/chapter2.2',
        'module2/chapter2.3',
      ],
      link: {
        type: 'doc',
        id: 'module2/intro',
      },
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module3/intro',
        'module3/chapter3.1',
        'module3/chapter3.2',
        'module3/chapter3.3',
      ],
      link: {
        type: 'doc',
        id: 'module3/intro',
      },
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) & Capstone',
      items: [
        'module4/intro',
        'module4/chapter4.1',
        'module4/chapter4.2',
        'module4/chapter4.3',
      ],
      link: {
        type: 'doc',
        id: 'module4/intro',
      },
    },
  ],
};

module.exports = sidebars;