import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
  shadow?: 'sm' | 'md' | 'lg';
}

const Card: React.FC<CardProps> = ({
  children,
  className = '',
  padding = 'md',
  shadow = 'sm'
}) => {
  const paddingClass = `p-${padding === 'sm' ? '3' : padding === 'lg' ? '5' : '4'}`;
  const shadowClass = `shadow-${shadow}`;
  
  const classes = [
    'bg-white',
    'border-radius-lg',
    paddingClass,
    shadowClass,
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={classes}>
      {children}
    </div>
  );
};

export default Card;