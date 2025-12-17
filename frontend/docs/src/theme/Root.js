import React, { useEffect } from 'react';
import RagChatWidget from '../components/RagChatWidget';

export default function Root({children}) {
  useEffect(() => {
    console.log('Root component loaded with RagChatWidget');
  }, []);

  return (
    <>
      {children}
      <RagChatWidget />
    </>
  );
}