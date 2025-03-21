// import js from '@eslint/js';
// import globals from 'globals';
// import reactHooks from 'eslint-plugin-react-hooks';
// import reactRefresh from 'eslint-plugin-react-refresh';
// import tseslint from 'typescript-eslint';
// import prettier from 'eslint-config-prettier'; // Prettier 설정 추가
// import prettierPlugin from 'eslint-plugin-prettier'; // Prettier 플러그인 추가

export default [
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      'eslint:recommended',
      'plugin:@typescript-eslint/recommended',
      'plugin:react-hooks/recommended',
      'prettier',
    ],
    plugins: ['@typescript-eslint', 'react-refresh', 'prettier'],
    rules: {
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      'prettier/prettier': ['error', { endOfLine: 'auto' }],
      '@typescript-eslint/no-unused-vars': 'off',
    },
    parserOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
    },
    env: {
      browser: true,
      es2020: true,
    },
    ignores: ['dist/**'],
  },
];