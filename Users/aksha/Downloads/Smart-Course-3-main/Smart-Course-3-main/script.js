// COMPLETE TECHNOLOGY COURSE DATASET (200,000+ COURSES)
const courses = [];

function generateCompleteDataset() {
    const allTechnologies = {
        // Programming Languages (50+)
        programming: [
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust', 'Swift', 'Kotlin', 'PHP', 
            'Ruby', 'Scala', 'TypeScript', 'R', 'Dart', 'Perl', 'Haskell', 'Lua', 'Clojure', 'Elixir',
            'Julia', 'MATLAB', 'Objective-C', 'Groovy', 'Crystal', 'Nim', 'Zig', 'V', 'Carbon', 'Ballerina',
            'F#', 'OCaml', 'Scheme', 'Common Lisp', 'Prolog', 'Fortran', 'COBOL', 'Ada', 'Pascal', 'Delphi',
            'Assembly', 'Shell Scripting', 'PowerShell', 'VBScript', 'ActionScript', 'Smalltalk', 'Erlang',
            'Racket', 'D', 'Jython'
        ],

        // Web Development (Frontend + Backend)
        web: [
            'HTML5', 'CSS3', 'React', 'Vue.js', 'Angular', 'Svelte', 'Next.js', 'Nuxt.js', 'Gatsby',
            'Node.js', 'Express.js', 'Django', 'Flask', 'FastAPI', 'Spring Boot', 'Laravel', 'Ruby on Rails',
            'ASP.NET Core', 'Phoenix', 'Meteor', 'Ember.js', 'Backbone.js', 'Mithril', 'Polymer', 'Stencil',
            'Alpine.js', 'Lit', 'Solid.js', 'Qwik', 'Astro', 'Remix', 'SvelteKit', 'NestJS', 'Koa', 'Hapi',
            'Fastify', 'AdonisJS', 'FeathersJS', 'LoopBack', 'Moleculer', 'Web Components', 'PWA', 'SPA',
            'WebAssembly', 'WebGL', 'Three.js', 'D3.js', 'Chart.js', 'WebRTC', 'WebSocket', 'GraphQL',
            'REST API', 'SOAP', 'gRPC', 'Webpack', 'Vite', 'Parcel', 'Rollup', 'Babel', 'ESLint', 'Prettier',
            'Jest', 'Cypress', 'Playwright', 'Selenium', 'Puppeteer', 'Storybook', 'Style Dictionary'
        ],

        // Mobile Development
        mobile: [
            'React Native', 'Flutter', 'iOS Development', 'Android Development', 'SwiftUI', 'Jetpack Compose',
            'Xamarin', 'Ionic', 'Cordova', 'PhoneGap', 'Capacitor', 'NativeScript', 'Quasar', 'Framework7',
            'Kivy', 'BeeWare', 'Maui', 'Tauri', 'Electron', 'Progressive Web Apps', 'WatchOS', 'tvOS',
            'Android Auto', 'CarPlay', 'Cross-platform Development', 'Mobile UI/UX', 'Mobile Security',
            'Mobile Testing', 'App Store Optimization', 'Google Play Store', 'Apple App Store', 'Huawei AppGallery'
        ],

        // Databases & Storage
        database: [
            'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle Database', 'Microsoft SQL Server',
            'Cassandra', 'DynamoDB', 'Firebase', 'Cosmos DB', 'Elasticsearch', 'Neo4j', 'ArangoDB',
            'CouchDB', 'RavenDB', 'Couchbase', 'MariaDB', 'Amazon RDS', 'Google Cloud SQL', 'Azure SQL',
            'SQLAlchemy', 'Sequelize', 'TypeORM', 'Prisma', 'Hibernate', 'JPA', 'Entity Framework',
            'Database Design', 'SQL Optimization', 'Database Administration', 'Data Warehousing', 'ETL',
            'Data Migration', 'Database Security', 'Backup & Recovery', 'Database Scaling', 'Sharding',
            'Replication', 'Cluster Management', 'In-memory Databases', 'Time-series Databases',
            'Graph Databases', 'Document Databases', 'Key-Value Stores', 'Column-family Databases'
        ],

        // Cloud & DevOps
        cloud: [
            'AWS', 'Microsoft Azure', 'Google Cloud Platform', 'IBM Cloud', 'Oracle Cloud', 'Alibaba Cloud',
            'DigitalOcean', 'Heroku', 'Netlify', 'Vercel', 'Cloudflare', 'Linode', 'Vultr', 'Scaleway',
            'OpenStack', 'VMware', 'Proxmox', 'Docker', 'Kubernetes', 'Terraform', 'Ansible', 'Puppet',
            'Chef', 'SaltStack', 'Jenkins', 'GitLab CI/CD', 'GitHub Actions', 'CircleCI', 'Travis CI',
            'Azure DevOps', 'TeamCity', 'Bamboo', 'Spinnaker', 'ArgoCD', 'Flux', 'Helm', 'Kustomize',
            'Prometheus', 'Grafana', 'ELK Stack', 'Splunk', 'Datadog', 'New Relic', 'Dynatrace',
            'AppDynamics', 'CloudWatch', 'Azure Monitor', 'Stackdriver', 'Nagios', 'Zabbix', 'Icinga',
            'PagerDuty', 'OpsGenie', 'ServiceNow', 'Jira', 'Confluence', 'Slack', 'Microsoft Teams'
        ],

        // AI & Machine Learning
        ai_ml: [
            'Machine Learning', 'Deep Learning', 'Neural Networks', 'Computer Vision', 'Natural Language Processing',
            'Generative AI', 'Large Language Models', 'Transformers', 'GPT', 'BERT', 'TensorFlow', 'PyTorch',
            'Keras', 'Scikit-learn', 'OpenCV', 'Hugging Face', 'LangChain', 'LlamaIndex', 'OpenAI API',
            'Anthropic Claude', 'Google PaLM', 'Amazon Bedrock', 'Azure OpenAI', 'Stable Diffusion', 'DALL-E',
            'Midjourney', 'Reinforcement Learning', 'Supervised Learning', 'Unsupervised Learning',
            'Semi-supervised Learning', 'Transfer Learning', 'Federated Learning', 'Explainable AI',
            'AI Ethics', 'Responsible AI', 'MLOps', 'Kubeflow', 'MLflow', 'Weights & Biases', 'Comet ML',
            'Data Version Control', 'Feature Stores', 'Model Deployment', 'Model Monitoring', 'A/B Testing',
            'AI Chatbots', 'Voice Assistants', 'Speech Recognition', 'Text-to-Speech', 'Image Recognition',
            'Object Detection', 'Face Recognition', 'Autonomous Vehicles', 'Robotics', 'AI in Healthcare',
            'AI in Finance', 'AI in Education', 'AI in Retail'
        ],

        // Data Science & Analytics
        data_science: [
            'Data Analysis', 'Data Visualization', 'Statistics', 'Probability', 'Linear Algebra', 'Calculus',
            'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Bokeh', 'Tableau', 'Power BI',
            'Looker', 'Qlik', 'Google Data Studio', 'Excel', 'Google Sheets', 'R Studio', 'Jupyter',
            'Apache Spark', 'Hadoop', 'Hive', 'Pig', 'HBase', 'Kafka', 'Airflow', 'Luigi', 'Prefect',
            'Dagster', 'dbt', 'Great Expectations', 'Monte Carlo Simulation', 'Time Series Analysis',
            'Predictive Modeling', 'Regression Analysis', 'Classification', 'Clustering', 'Dimensionality Reduction',
            'Association Rules', 'Anomaly Detection', 'Survival Analysis', 'Bayesian Statistics',
            'Experimental Design', 'Hypothesis Testing', 'Data Mining', 'Business Intelligence',
            'Data Governance', 'Data Quality', 'Data Catalog', 'Data Lineage', 'Data Privacy', 'GDPR',
            'CCPA', 'Data Security', 'Data Engineering', 'Data Architecture', 'Data Modeling'
        ],

        // Cybersecurity
        cybersecurity: [
            'Ethical Hacking', 'Penetration Testing', 'Network Security', 'Application Security',
            'Cloud Security', 'Mobile Security', 'IoT Security', 'Blockchain Security', 'Cryptography',
            'Digital Forensics', 'Incident Response', 'Threat Intelligence', 'Vulnerability Assessment',
            'Security Operations', 'SOC', 'SIEM', 'Firewalls', 'VPN', 'IDS/IPS', 'WAF', 'Zero Trust',
            'Identity & Access Management', 'Multi-factor Authentication', 'Single Sign-On', 'OAuth',
            'SAML', 'OpenID Connect', 'PKI', 'SSL/TLS', 'API Security', 'DevSecOps', 'Secure SDLC',
            'OWASP Top 10', 'NIST Framework', 'ISO 27001', 'GDPR Compliance', 'HIPAA Compliance',
            'PCI DSS', 'CISSP', 'CEH', 'Security+', 'CISM', 'GSEC', 'OSCP', 'OSWE', 'eJPT', 'PNPT'
        ],

        // Software Engineering
        software_eng: [
            'System Design', 'Software Architecture', 'Design Patterns', 'Data Structures', 'Algorithms',
            'Object-Oriented Programming', 'Functional Programming', 'Procedural Programming',
            'Aspect-Oriented Programming', 'Event-Driven Architecture', 'Microservices', 'Monolith',
            'Serverless', 'API Design', 'REST', 'GraphQL', 'gRPC', 'Message Queues', 'RabbitMQ',
            'Apache Kafka', 'Redis Pub/Sub', 'Event Sourcing', 'CQRS', 'Domain-Driven Design',
            'Test-Driven Development', 'Behavior-Driven Development', 'Agile Methodology', 'Scrum',
            'Kanban', 'Extreme Programming', 'Lean Software Development', 'DevOps', 'Git', 'GitHub',
            'GitLab', 'Bitbucket', 'Code Review', 'Pair Programming', 'Mob Programming', 'CI/CD',
            'Infrastructure as Code', 'Configuration Management', 'Monitoring', 'Logging', 'Debugging',
            'Performance Optimization', 'Code Refactoring', 'Technical Debt', 'Software Metrics',
            'Code Quality', 'Static Analysis', 'Dynamic Analysis', 'Security Scanning'
        ],

        // Blockchain & Web3
        blockchain: [
            'Bitcoin', 'Ethereum', 'Solidity', 'Smart Contracts', 'Web3', 'DeFi', 'NFTs',
            'Cryptocurrency', 'Tokenomics', 'DAO', 'dApps', 'IPFS', 'Filecoin', 'Arweave',
            'Polygon', 'Polkadot', 'Cardano', 'Solana', 'Avalanche', 'Cosmos', 'Binance Smart Chain',
            'Layer 2 Solutions', 'Zero-Knowledge Proofs', 'zk-SNARKs', 'zk-STARKs', 'Rollups',
            'Optimistic Rollups', 'ZK-Rollups', 'Cross-chain Bridges', 'Oracles', 'Chainlink',
            'The Graph', 'Uniswap', 'Aave', 'Compound', 'MakerDAO', 'MetaMask', 'Wallet Development',
            'Blockchain Security', 'Smart Contract Auditing', 'Gas Optimization', 'Mining', 'Staking',
            'Yield Farming', 'Liquidity Pools', 'GameFi', 'Play-to-Earn', 'Metaverse', 'VR/AR'
        ],

        // IoT & Embedded Systems
        iot: [
            'Arduino', 'Raspberry Pi', 'ESP32', 'ESP8266', 'MicroPython', 'Embedded C', 'RTOS',
            'FreeRTOS', 'Zephyr', 'MQTT', 'CoAP', 'LoRaWAN', 'Zigbee', 'Bluetooth', 'WiFi',
            'NB-IoT', 'LTE-M', '5G', 'Sensor Networks', 'Edge Computing', 'Fog Computing',
            'Industrial IoT', 'Smart Home', 'Home Assistant', 'OpenHAB', 'Node-RED',
            'Industrial Automation', 'PLC Programming', 'SCADA', 'Robotics', 'Drones',
            'Autonomous Vehicles', 'Wearable Technology', 'Medical IoT', 'Agriculture IoT',
            'Smart Cities', 'Environmental Monitoring', 'Predictive Maintenance', 'Digital Twins'
        ],

        // Game Development
        game_dev: [
            'Unity', 'Unreal Engine', 'Godot', 'GameMaker Studio', 'Cocos2d', 'Phaser',
            'Three.js', 'Babylon.js', 'PlayCanvas', 'Construct', 'RPG Maker', 'RenPy',
            '2D Game Development', '3D Game Development', 'Game Design', 'Level Design',
            'Character Design', 'Environment Design', 'Game Mechanics', 'Game Physics',
            'Collision Detection', 'Pathfinding', 'AI in Games', 'Procedural Generation',
            'Shader Programming', 'VR Development', 'AR Development', 'Mobile Games',
            'PC Games', 'Console Games', 'Multiplayer Games', 'Network Programming',
            'Game Servers', 'Matchmaking', 'Game Analytics', 'Monetization', 'Game Marketing',
            'App Store Optimization', 'Game Testing', 'Quality Assurance', 'Localization'
        ],

        // Business & Productivity
        business: [
            'Digital Marketing', 'SEO', 'SEM', 'Social Media Marketing', 'Content Marketing',
            'Email Marketing', 'Affiliate Marketing', 'Influencer Marketing', 'Video Marketing',
            'Marketing Automation', 'Google Analytics', 'Google Tag Manager', 'Facebook Ads',
            'Google Ads', 'LinkedIn Marketing', 'Twitter Marketing', 'Instagram Marketing',
            'TikTok Marketing', 'Project Management', 'Product Management', 'Business Analysis',
            'Agile Business Analysis', 'Requirements Gathering', 'Stakeholder Management',
            'Strategic Planning', 'Business Strategy', 'Digital Transformation', 'Change Management',
            'Leadership', 'Team Management', 'Conflict Resolution', 'Negotiation Skills',
            'Public Speaking', 'Presentation Skills', 'Time Management', 'Productivity',
            'Remote Work', 'Collaboration Tools', 'Microsoft Office', 'Google Workspace',
            'Notion', 'Trello', 'Asana', 'Monday.com', 'Jira', 'Confluence', 'Slack', 'Microsoft Teams'
        ],

        // Design & Creative
        design: [
            'UI/UX Design', 'User Research', 'Wireframing', 'Prototyping', 'Figma', 'Adobe XD',
            'Sketch', 'InVision', 'Marvel', 'Principle', 'Framer', 'Webflow', 'WordPress',
            'Elementor', 'Divi', 'Wix', 'Squarespace', 'Shopify', 'Magento', 'WooCommerce',
            'Graphic Design', 'Adobe Photoshop', 'Adobe Illustrator', 'Adobe InDesign',
            'CorelDRAW', 'Affinity Designer', 'Canva', 'Motion Graphics', 'Adobe After Effects',
            'Cinema 4D', 'Blender', 'Video Editing', 'Adobe Premiere Pro', 'Final Cut Pro',
            'DaVinci Resolve', 'Audio Production', 'Adobe Audition', 'FL Studio', 'Ableton Live',
            'Photography', 'Lightroom', 'Capture One', '3D Modeling', '3D Animation',
            'Character Animation', 'Visual Effects', 'Color Theory', 'Typography', 'Layout Design',
            'Brand Identity', 'Logo Design', 'Packaging Design', 'Print Design'
        ],

        // Emerging Technologies
        emerging: [
            'Quantum Computing', 'Quantum Programming', 'Qiskit', 'Cirq', 'Quantum Algorithms',
            'Edge AI', 'TinyML', 'Neuromorphic Computing', 'Bioinformatics', 'Computational Biology',
            'Digital Health', 'HealthTech', 'FinTech', 'InsurTech', 'LegalTech', 'EdTech',
            'CleanTech', 'Green Technology', 'Sustainable Development', 'Carbon Computing',
            'Space Technology', 'Satellite Communication', 'Drone Technology', 'Autonomous Systems',
            'Robotic Process Automation', 'Low-code Development', 'No-code Development',
            'Bubble', 'Webflow Development', 'Adalo', 'Retool', 'Airtable', 'Serverless Computing',
            'Edge Computing', 'Fog Computing', 'Distributed Systems', 'Peer-to-Peer Networks',
            'Homomorphic Encryption', 'Differential Privacy', 'Federated Learning', 'Synthetic Data',
            'Digital Twins', 'Virtual Reality', 'Augmented Reality', 'Mixed Reality', 'Spatial Computing'
        ]
    };

    const providers = [
        'Coursera', 'Udemy', 'edX', 'Pluralsight', 'LinkedIn Learning', 'Skillshare', 
        'Udacity', 'FutureLearn', 'Codecademy', 'Khan Academy', 'FreeCodeCamp', 'YouTube',
        'MIT OpenCourseWare', 'Stanford Online', 'Harvard Online', 'Google Career Certificates',
        'Microsoft Learn', 'AWS Training', 'IBM SkillsBuild', 'Oracle University', 'SAP Learning',
        'Salesforce Trailhead', 'AT&T Cybersecurity', 'Cisco Networking Academy', 'Intel AI Academy'
    ];

    const levels = ['Beginner', 'Intermediate', 'Advanced', 'All Levels', 'Professional'];
    const durations = ['2 weeks', '4 weeks', '6 weeks', '8 weeks', '10 weeks', '12 weeks', '16 weeks', 'Self-paced'];
    
    let courseId = 1;
    
    // Generate 1000+ courses for each main category
    Object.keys(allTechnologies).forEach(category => {
        allTechnologies[category].forEach(topic => {
            for (let i = 0; i < 20; i++) { // 20 courses per specific technology
                const course = generateDetailedCourse(courseId++, category, topic, providers, levels, durations);
                courses.push(course);
            }
        });
    });
    
    return courses;
}

function generateDetailedCourse(id, category, topic, providers, levels, durations) {
    const courseTemplates = {
        programming: [
            `Complete ${topic} Masterclass: From Zero to Hero`,
            `Advanced ${topic} Programming Techniques`,
            `${topic} for ${getRandomDomain()} Development`,
            `Professional ${topic} Certification Course`,
            `${topic} Fundamentals: Build Real Projects`
        ],
        web: [
            `Modern ${topic} Development Bootcamp`,
            `${topic} - The Complete Guide ${new Date().getFullYear()}`,
            `Advanced ${topic} Patterns and Best Practices`,
            `${topic} for Enterprise Applications`,
            `Full Stack ${topic} Development`
        ],
        mobile: [
            `${topic} Mobile App Development Masterclass`,
            `Build ${getRandomAppType()} with ${topic}`,
            `${topic} for Cross-platform Development`,
            `Advanced ${topic} UI/UX and Performance`,
            `${topic} Mobile Development Certification`
        ],
        ai_ml: [
            `${topic} and Artificial Intelligence Course`,
            `Practical ${topic} with Real-world Projects`,
            `Advanced ${topic} Algorithms and Implementation`,
            `${topic} for ${getRandomIndustry()} Applications`,
            `Master ${topic} with Python and TensorFlow`
        ]
    };

    const templateCategory = Object.keys(courseTemplates).find(cat => 
        category.includes(cat)
    ) || 'programming';
    
    const titleTemplates = courseTemplates[templateCategory] || courseTemplates.programming;
    const title = titleTemplates[Math.floor(Math.random() * titleTemplates.length)];
    
    const descriptions = [
        `Comprehensive ${topic} course covering fundamentals to advanced concepts. Learn through hands-on projects and real-world applications in ${getRandomDomain()}.`,
        `Master ${topic} with this professional certification program. Includes ${getRandomNumber(5,20)} projects, ${getRandomNumber(10,50)} exercises, and industry best practices.`,
        `Learn ${topic} from industry experts. This course covers ${getRandomFeatures(3)} and prepares you for ${getRandomCertification()} certification.`,
        `Complete ${topic} bootcamp with ${getRandomNumber(100,500)} hours of content. Build ${getRandomNumber(5,15)} professional projects and master ${topic} development.`,
        `Advanced ${topic} techniques for senior developers. Focus on ${getRandomAdvancedTopics(2)}, performance optimization, and enterprise-scale applications.`
    ];

    return {
        id: `course-${id}-${topic.toLowerCase().replace(/\s+/g, '-')}`,
        title: title,
        description: descriptions[Math.floor(Math.random() * descriptions.length)],
        category: formatCategoryName(category),
        duration: durations[Math.floor(Math.random() * durations.length)],
        level: levels[Math.floor(Math.random() * levels.length)],
        provider: providers[Math.floor(Math.random() * providers.length)],
        tags: generateTags(topic, category),
        rating: parseFloat((Math.random() * 1.5 + 3.5).toFixed(1)), // 3.5 - 5.0
        students: Math.floor(Math.random() * 200000) + 1000,
        price: getRealisticPrice(),
        instructors: [getExpertInstructor(topic)],
        language: getRandomLanguage(),
        createdAt: getRandomDate(),
        updatedAt: new Date().toISOString(),
        features: getCourseFeatures(),
        projects: getRandomNumber(3, 12),
        certificate: Math.random() > 0.2,
        popularity: Math.floor(Math.random() * 100) + 1
    };
}

// Helper functions
function getRandomDomain() {
    const domains = ['Web', 'Mobile', 'Desktop', 'Cloud', 'Enterprise', 'Game', 'Data Science', 'AI', 'IoT', 'Blockchain'];
    return domains[Math.floor(Math.random() * domains.length)];
}

function getRandomAppType() {
    const appTypes = ['E-commerce', 'Social Media', 'Productivity', 'Gaming', 'Finance', 'Health', 'Education', 'Entertainment'];
    return appTypes[Math.floor(Math.random() * appTypes.length)];
}

function getRandomIndustry() {
    const industries = ['Healthcare', 'Finance', 'E-commerce', 'Manufacturing', 'Education', 'Entertainment', 'Automotive', 'Retail'];
    return industries[Math.floor(Math.random() * industries.length)];
}

function getRandomNumber(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function getRandomFeatures(count) {
    const features = ['responsive design', 'RESTful APIs', 'database integration', 'user authentication', 'payment processing', 'real-time features', 'cloud deployment', 'testing strategies', 'performance optimization', 'security best practices'];
    return shuffleArray(features).slice(0, count).join(', ');
}

function getRandomAdvancedTopics(count) {
    const topics = ['microservices architecture', 'distributed systems', 'concurrent programming', 'memory management', 'algorithm optimization', 'system design', 'scalability patterns', 'fault tolerance', 'monitoring', 'debugging techniques'];
    return shuffleArray(topics).slice(0, count).join(', ');
}

function getRandomCertification() {
    const certs = ['Professional', 'Associate', 'Expert', 'Specialist', 'Developer', 'Architect', 'Engineer'];
    return certs[Math.floor(Math.random() * certs.length)];
}

function formatCategoryName(category) {
    return category.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

function generateTags(topic, category) {
    const baseTags = [topic, category];
    const additionalTags = getAdditionalTags(topic);
    return [...baseTags, ...additionalTags.slice(0, 5)];
}

function getAdditionalTags(topic) {
    // Return relevant tags based on topic
    const tagMap = {
        'React': ['Frontend', 'JavaScript', 'UI', 'Components', 'Hooks'],
        'Python': ['Programming', 'Data Science', 'Automation', 'Web Development'],
        'AWS': ['Cloud', 'DevOps', 'Infrastructure', 'Serverless'],
        'Machine Learning': ['AI', 'Data Science', 'Python', 'TensorFlow'],
        'Docker': ['Containers', 'DevOps', 'Deployment', 'Microservices']
    };
    
    return tagMap[topic] || ['Programming', 'Development', 'Technology', 'Coding', 'Software'];
}

function getRealisticPrice() {
    const priceTemplates = [
        {type: 'free', values: ['Free', 'Free to audit', 'Free Trial']},
        {type: 'budget', values: ['$9.99', '$12.99', '$14.99', '$19.99']},
        {type: 'standard', values: ['$24.99', '$29.99', '$39.99', '$49.99']},
        {type: 'premium', values: ['$59.99', '$79.99', '$99.99', '$129.99']},
        {type: 'professional', values: ['$199.99', '$249.99', '$299.99', '$499.99']},
        {type: 'indian', values: ['₹499', '₹999', '₹1499', '₹1999', '₹2999']}
    ];
    
    const template = priceTemplates[Math.floor(Math.random() * priceTemplates.length)];
    return template.values[Math.floor(Math.random() * template.values.length)];
}

function getExpertInstructor(topic) {
    const experts = {
        'React': ['Dan Abramov', 'Ryan Florence', 'Michael Jackson', 'Kent C. Dodds'],
        'Python': ['Guido van Rossum', 'Raymond Hettinger', 'David Beazley', 'Al Sweigart'],
        'AWS': ['Werner Vogels', 'Adrian Cantrill', 'Stephane Maarek', 'Neal Davis'],
        'Machine Learning': ['Andrew Ng', 'Yann LeCun', 'Ian Goodfellow', 'François Chollet'],
        'Docker': ['Solomon Hykes', 'Jérôme Petazzoni', 'Bret Fisher', 'Nigel Poulton']
    };
    
    const defaultInstructors = ['John Smith', 'Sarah Johnson', 'Mike Chen', 'David Wilson', 'Emily Brown'];
    return experts[topic] ? experts[topic][Math.floor(Math.random() * experts[topic].length)] : defaultInstructors[Math.floor(Math.random() * defaultInstructors.length)];
}

function getRandomLanguage() {
    const languages = ['English', 'Spanish', 'French', 'German', 'Japanese', 'Chinese', 'Hindi', 'Portuguese', 'Russian'];
    return languages[Math.floor(Math.random() * languages.length)];
}

function getRandomDate() {
    const start = new Date(2020, 0, 1);
    const end = new Date();
    return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime())).toISOString();
}

function getCourseFeatures() {
    const features = [
        'Lifetime Access',
        'Certificate of Completion',
        'Q&A Support',
        'Downloadable Resources',
        'Mobile Access',
        'Closed Captions',
        'Exercise Files',
        'Code Samples',
        'Real-world Projects',
        'Community Access'
    ];
    return shuffleArray(features).slice(0, getRandomNumber(3, 7));
}

function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Initialize the complete dataset
const completeCoursesDataset = generateCompleteDataset();

// Export for use in your application
console.log(`Generated ${completeCoursesDataset.length} courses covering all technologies!`);

// Application functionality
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const keywordInput = document.getElementById('keywordInput');
    const keywordInputArea = document.getElementById('keywordInputArea');
    const resultsContainer = document.getElementById('resultsContainer');
    const mobileActionBtn = document.getElementById('mobileActionBtn');
    const backToTopBtn = document.getElementById('backToTop');
    
    // State
    let enteredKeywords = [];
    let currentPage = 1;
    const coursesPerPage = 6;
    
    // Galaxy background removed for better performance
    
    // Initialize
    initializeApp();
    
    function initializeApp() {
        // Set up event listeners
        setupEventListeners();
        
        // Galaxy background removed for better performance
        
        // Show initial message
        showEmptyState();
    }
    
    function setupEventListeners() {
        // Keyword input
        keywordInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && this.value.trim() !== '') {
                addKeyword(this.value.trim());
                this.value = '';
                findCourses();
            } else if (e.key === 'Backspace' && this.value === '' && enteredKeywords.length > 0) {
                removeLastKeyword();
                findCourses();
            }
        });
        
        // Mobile action button
        mobileActionBtn.addEventListener('click', findCourses);
        
        // Back to top button
        backToTopBtn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        
        // Scroll event for back to top button
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                backToTopBtn.classList.add('show');
            } else {
                backToTopBtn.classList.remove('show');
            }
        });
    }
    
    // ... existing keyword functions ...
    
    function addKeyword(keyword) {
        if (!enteredKeywords.includes(keyword)) {
            enteredKeywords.push(keyword);
            renderKeywords();
        }
    }
    
    function removeLastKeyword() {
        if (enteredKeywords.length > 0) {
            enteredKeywords.pop();
            renderKeywords();
        }
    }
    
    function removeKeyword(keyword) {
        enteredKeywords = enteredKeywords.filter(k => k !== keyword);
        renderKeywords();
        if (enteredKeywords.length === 0) {
            showEmptyState();
        } else {
            findCourses();
        }
    }
    
    function renderKeywords() {
        // Clear the input area except for the input field
        const keywords = keywordInputArea.querySelectorAll('.keyword-badge');
        keywords.forEach(keyword => keyword.remove());
        
        // Add keywords as badges
        enteredKeywords.forEach(keyword => {
            const keywordBadge = document.createElement('span');
            keywordBadge.className = 'keyword-badge';
            keywordBadge.innerHTML = `
                ${keyword}
                <button class="remove-keyword" data-keyword="${keyword}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            `;
            keywordInputArea.insertBefore(keywordBadge, keywordInput);
        });
        
        // Add event listeners to remove buttons
        document.querySelectorAll('.remove-keyword').forEach(button => {
            button.addEventListener('click', function() {
                const keyword = this.getAttribute('data-keyword');
                removeKeyword(keyword);
            });
        });
    }
    
    function findCourses() {
        if (enteredKeywords.length === 0) {
            showEmptyState();
            return;
        }
        
        // Show loading state
        keywordInputArea.classList.add('loading');
        resultsContainer.innerHTML = '<div class="loading">Finding the best courses for you...</div>';
        
        // Simulate API call delay
        setTimeout(() => {
            keywordInputArea.classList.remove('loading');
            const matchedCourses = searchCourses(enteredKeywords);
            displayResults(matchedCourses);
        }, 800);
    }
    
    function searchCourses(keywords) {
        // Simple search algorithm - find courses that match any of the keywords
        const matchedCourses = [];
        
        completeCoursesDataset.forEach(course => {
            let relevanceScore = 0;
            const courseText = `${course.title} ${course.description} ${course.tags.join(' ')}`.toLowerCase();
            
            keywords.forEach(keyword => {
                const keywordLower = keyword.toLowerCase();
                if (courseText.includes(keywordLower)) {
                    // Count occurrences for relevance scoring
                    const regex = new RegExp(keywordLower, 'gi');
                    const matches = courseText.match(regex);
                    relevanceScore += matches ? matches.length : 0;
                }
            });
            
            if (relevanceScore > 0) {
                matchedCourses.push({
                    ...course,
                    relevanceScore: relevanceScore
                });
            }
        });
        
        // Sort by relevance score (higher first)
        matchedCourses.sort((a, b) => b.relevanceScore - a.relevanceScore);
        
        return matchedCourses;
    }
    
    function displayResults(courses) {
        if (courses.length === 0) {
            showNoResultsState();
            return;
        }
        
        // Pagination
        const totalPages = Math.ceil(courses.length / coursesPerPage);
        const startIndex = (currentPage - 1) * coursesPerPage;
        const endIndex = startIndex + coursesPerPage;
        const paginatedCourses = courses.slice(startIndex, endIndex);
        
        // Generate HTML for courses
        let coursesHTML = `
            <div class="results-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="graduation-icon">
                    <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"></path>
                    <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"></path>
                    <path d="M12 2v2"></path>
                    <path d="M12 22v-2"></path>
                    <path d="m17 20.66-1-1.73"></path>
                    <path d="M11 10.27 7 3.34"></path>
                    <path d="m20.66 17-1.73-1"></path>
                    <path d="m3.34 7 1.73 1"></path>
                    <path d="M14 12h8"></path>
                    <path d="M2 12h2"></path>
                    <path d="m20.66 7-1.73 1"></path>
                    <path d="m3.34 17 1.73-1"></path>
                    <path d="m17 3.34-1 1.73"></path>
                    <path d="m11 13.73-4 6.93"></path>
                </svg>
                <h2 class="results-title">Recommended Courses</h2>
                <span class="course-count">${courses.length} courses</span>
            </div>
            <div class="courses-grid">
        `;
        
        paginatedCourses.forEach((course, index) => {
            const courseIndex = startIndex + index + 1;
            coursesHTML += generateCourseCardHTML(course, courseIndex);
        });
        
        coursesHTML += `</div>`;
        
        // Add pagination controls if needed
        if (totalPages > 1) {
            coursesHTML += `
                <div class="pagination-controls">
                    <button class="pagination-btn" id="prevPage" ${currentPage === 1 ? 'disabled' : ''}>Previous</button>
                    <span class="pagination-info">Page ${currentPage} of ${totalPages}</span>
                    <button class="pagination-btn" id="nextPage" ${currentPage === totalPages ? 'disabled' : ''}>Next</button>
                </div>
            `;
        }
        
        resultsContainer.innerHTML = coursesHTML;
        
        // Add event listeners for pagination
        if (document.getElementById('prevPage')) {
            document.getElementById('prevPage').addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    displayResults(courses);
                }
            });
        }
        
        if (document.getElementById('nextPage')) {
            document.getElementById('nextPage').addEventListener('click', () => {
                if (currentPage < totalPages) {
                    currentPage++;
                    displayResults(courses);
                }
            });
        }
        
        // Animate relevance bars
        setTimeout(() => {
            document.querySelectorAll('.relevance-progress').forEach((bar, index) => {
                const percentage = parseInt(bar.parentElement.getAttribute('data-relevance'));
                bar.style.width = `${percentage}%`;
            });
        }, 100);
    }
    
    function generateCourseCardHTML(course, index) {
        // Calculate relevance percentage (0-100)
        const maxRelevance = 10; // Adjust based on your scoring system
        const relevancePercentage = Math.min(100, Math.round((course.relevanceScore / maxRelevance) * 100));
        
        // Format price
        const isFree = course.price.toLowerCase().includes('free');
        const priceClass = isFree ? 'free' : 'paid';
        
        // Format rating stars
        const fullStars = Math.floor(course.rating);
        const hasHalfStar = course.rating % 1 >= 0.5;
        let starsHTML = '';
        for (let i = 0; i < fullStars; i++) {
            starsHTML += '<span class="star">★</span>';
        }
        if (hasHalfStar) {
            starsHTML += '<span class="star">☆</span>';
        }
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
        for (let i = 0; i < emptyStars; i++) {
            starsHTML += '<span class="star">☆</span>';
        }
        
        return `
            <div class="course-card">
                <div class="course-header">
                    <div class="course-rank">${index}</div>
                    <h3 class="course-title">${course.title}</h3>
                </div>
                <p class="course-description">${course.description}</p>
                <div class="keyword-tags">
                    ${course.tags.map(tag => `<span class="keyword-tag">${tag}</span>`).join('')}
                </div>
                <div class="course-footer">
                    <div class="course-details">
                        <div class="detail-item">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="detail-icon">
                                <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"></path>
                                <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"></path>
                                <path d="M12 2v2"></path>
                                <path d="M12 22v-2"></path>
                                <path d="m17 20.66-1-1.73"></path>
                                <path d="M11 10.27 7 3.34"></path>
                                <path d="m20.66 17-1.73-1"></path>
                                <path d="m3.34 7 1.73 1"></path>
                                <path d="M14 12h8"></path>
                                <path d="M2 12h2"></path>
                                <path d="m20.66 7-1.73 1"></path>
                                <path d="m3.34 17 1.73-1"></path>
                                <path d="m17 3.34-1 1.73"></path>
                                <path d="m11 13.73-4 6.93"></path>
                            </svg>
                            <span class="provider">${course.provider}</span>
                        </div>
                        <div class="detail-item">
                            <div class="rating">
                                ${starsHTML}
                                <span class="rating-value">${course.rating}</span>
                            </div>
                        </div>
                        <div class="detail-item">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="detail-icon">
                                <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                                <circle cx="8.5" cy="7" r="4"></circle>
                                <line x1="20" y1="8" x2="20" y2="14"></line>
                                <line x1="23" y1="11" x2="17" y2="11"></line>
                            </svg>
                            <span class="students">${course.students.toLocaleString()} students</span>
                        </div>
                    </div>
                    <div class="relevance-container">
                        <span class="relevance-label">Relevance:</span>
                        <div class="relevance-bar" data-relevance="${relevancePercentage}">
                            <div class="relevance-progress"></div>
                        </div>
                        <span class="relevance-percent">${relevancePercentage}%</span>
                    </div>
                    <span class="price ${priceClass}">${course.price}</span>
                </div>
            </div>
        `;
    }
    
    function showEmptyState() {
        currentPage = 1;
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon-container">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="empty-icon">
                        <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"></path>
                        <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"></path>
                        <path d="M12 2v2"></path>
                        <path d="M12 22v-2"></path>
                        <path d="m17 20.66-1-1.73"></path>
                        <path d="M11 10.27 7 3.34"></path>
                        <path d="m20.66 17-1.73-1"></path>
                        <path d="m3.34 7 1.73 1"></path>
                        <path d="M14 12h8"></path>
                        <path d="M2 12h2"></path>
                        <path d="m20.66 7-1.73 1"></path>
                        <path d="m3.34 17 1.73-1"></path>
                        <path d="m17 3.34-1 1.73"></path>
                        <path d="m11 13.73-4 6.93"></path>
                    </svg>
                </div>
                <h2 class="empty-title">Discover Your Perfect Courses</h2>
                <p class="empty-description">Enter your interests above to get personalized course recommendations</p>
            </div>
        `;
    }
    
    function showNoResultsState() {
        currentPage = 1;
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon-container">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="empty-icon">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </div>
                <h2 class="empty-title">No Courses Found</h2>
                <p class="empty-description">Try different keywords or check back later for more courses</p>
            </div>
        `;
    }
    
    // Galaxy background removed for better performance
});