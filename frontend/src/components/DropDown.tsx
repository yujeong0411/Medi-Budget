import React, { useState, useRef, useEffect } from 'react';

interface Option {
  [key: string]: any; // 인덱스 시그니처 추가
  id: number;
  name: string;
}

interface DropdownProps {
  id: string;
  options: Option[];
  placeholder: string;
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  loading?: boolean;
  error?: string | null;
  keyProperty: string;
  displayProperty: string;
}

const SearchableDropdown: React.FC<DropdownProps> = ({
                                                       id,
                                                       options,
                                                       placeholder,
                                                       label,
                                                       value,
                                                       onChange,
                                                       loading = false,
                                                       error = '',
                                                       keyProperty,
                                                       displayProperty,
                                                     }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const optionsRef = useRef<(HTMLDivElement | null)[]>([]);

  // 선택된 옵션의 표시 이름 가져오기
  const getSelectedName = () => {
    if (!value) return '';
    const selectedOption = options.find(
      (option) => option[keyProperty as keyof typeof option] === value
    );
    return selectedOption ? String(selectedOption[displayProperty]) : '';
  };

  // 컴포넌트 마운트 또는 value 변경 시 inputValue 업데이트
  useEffect(() => {
    if (!isSearchMode) {
      setInputValue(getSelectedName());
    }
  }, [value, isSearchMode]);

  // 검색어에 따라 옵션 필터링
  const filteredOptions = options.filter((option) =>
    String(option[displayProperty as keyof typeof option])
      .toLowerCase()
      .includes(inputValue.toLowerCase())
  );

  // 외부 클릭 감지하여 드롭다운 닫기
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setIsSearchMode(false);
        setInputValue(getSelectedName()); // 선택된 값으로 복원
        setActiveIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [value]);

  // 필터링된 옵션이 변경될 때 activeIndex 재설정
  useEffect(() => {
    if (isOpen && filteredOptions.length > 0) {
      // 현재 선택된 값의 인덱스를 찾음
      const selectedIndex = filteredOptions.findIndex(
        option => option[keyProperty as keyof typeof option] === value
      );
      setActiveIndex(selectedIndex !== -1 ? selectedIndex : 0);
    }
  }, [filteredOptions.length, isOpen]);

  // 활성화된 옵션을 화면에 보이게 스크롤
  useEffect(() => {
    if (isOpen && activeIndex >= 0 && optionsRef.current[activeIndex]) {
      optionsRef.current[activeIndex]?.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIndex, isOpen]);

  // 드롭다운 열기/닫기 토글
  const toggleDropdown = () => {
    const newIsOpen = !isOpen;
    setIsOpen(newIsOpen);

    if (newIsOpen) {
      // 드롭다운 열릴 때 인풋에 포커스
      inputRef.current?.focus();

      // 활성 인덱스 설정
      if (filteredOptions.length > 0) {
        const selectedIndex = filteredOptions.findIndex(
          option => option[keyProperty as keyof typeof option] === value
        );
        setActiveIndex(selectedIndex !== -1 ? selectedIndex : 0);
      }
    }
  };

  // 옵션 선택 핸들러
  const handleOptionSelect = (optionId: number) => {
    onChange(optionId);
    setIsOpen(false);
    setIsSearchMode(false);
    setActiveIndex(-1);

    // 새로 선택된 옵션의 이름으로 입력 값 업데이트
    const selectedOption = options.find(
      (option) => option[keyProperty as keyof typeof option] === optionId
    );
    if (selectedOption) {
      setInputValue(String(selectedOption[displayProperty]));
    }

    // 다시 입력 필드에 포커스
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  // 검색어 변경 핸들러
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    setIsSearchMode(true);

    if (!isOpen) {
      setIsOpen(true);
    }

    // 입력값이 비어있으면 선택 값도 초기화
    if (newValue === '') {
      onChange(null);
    }

    // 검색어 변경 시 첫 번째 항목으로 활성 인덱스 재설정
    if (filteredOptions.length > 0) {
      setActiveIndex(0);
    }
  };

  // 입력 필드 클릭 핸들러
  const handleInputClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsSearchMode(true);
    if (!isOpen) {
      setIsOpen(true);
      if (filteredOptions.length > 0) {
        setActiveIndex(0);
      }
    }
  };

  // 키보드 네비게이션 핸들러
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // 드롭다운이 닫혀있고 아래 화살표나 Enter를 누르면 드롭다운 열기
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        e.preventDefault();
        setIsOpen(true);
        if (filteredOptions.length > 0) {
          setActiveIndex(0);
        }
      }
      return;
    }

    // 드롭다운이 열려있을 때 키 처리
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault(); // 스크롤 방지를 위해 기본 동작 취소
        if (filteredOptions.length > 0) {
          setActiveIndex(prev =>
            prev < filteredOptions.length - 1 ? prev + 1 : 0
          );
        }
        break;

      case 'ArrowUp':
        e.preventDefault(); // 스크롤 방지를 위해 기본 동작 취소
        if (filteredOptions.length > 0) {
          setActiveIndex(prev =>
            prev > 0 ? prev - 1 : filteredOptions.length - 1
          );
        }
        break;

      case 'Enter':
        e.preventDefault();
        if (activeIndex >= 0 && activeIndex < filteredOptions.length) {
          const selectedOption = filteredOptions[activeIndex];
          handleOptionSelect(selectedOption[keyProperty as keyof typeof selectedOption] as number);
        }
        break;

      case 'Escape':
        e.preventDefault();
        setIsOpen(false);
        setIsSearchMode(false);
        setInputValue(getSelectedName());
        setActiveIndex(-1);
        break;

      case 'Tab':
        // Tab 키는 기본 동작 유지 (다음 요소로 이동)
        setIsOpen(false);
        setIsSearchMode(false);
        setInputValue(getSelectedName());
        setActiveIndex(-1);
        break;
    }
  };

  return (
    <div className="flex flex-col" ref={dropdownRef}>
      <label htmlFor={id} className="text-base sm:text-lg mb-1 sm:mb-2">
        {label}
      </label>
      <div className="relative">
        <div
          className="p-3 rounded-lg min-h-[50px] bg-white flex items-center justify-between cursor-pointer border border-gray-300"
          onClick={toggleDropdown}
        >
          <input
            id={id}
            ref={inputRef}
            type="text"
            className="w-full outline-none"
            placeholder={placeholder}
            value={inputValue}
            onChange={handleSearchChange}
            onClick={handleInputClick}
            onKeyDown={handleKeyDown}
            disabled={loading}
            autoComplete="off"
            role="combobox"
            aria-expanded={isOpen}
            aria-controls={`${id}-listbox`}
            aria-autocomplete="list"
            aria-activedescendant={activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined}
          />
          <svg
            className={`w-5 h-5 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </div>

        {isOpen && (
          <div
            className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto"
            role="listbox"
            id={`${id}-listbox`}
          >
            {loading ? (
              <div className="p-3 text-center text-gray-500">로딩 중...</div>
            ) : filteredOptions.length === 0 ? (
              <div className="p-3 text-center text-gray-500">검색 결과가 없습니다</div>
            ) : (
              filteredOptions.map((option, index) => (
                <div
                  ref={el => { optionsRef.current[index] = el; }}
                  key={option[keyProperty as keyof typeof option] as number}
                  className={`p-3 hover:bg-gray-100 cursor-pointer ${
                    index === activeIndex ? 'bg-indigo-100 font-medium' : ''
                  } ${
                    value === option[keyProperty as keyof typeof option]
                      ? 'bg-indigo-50 text-indigo-600'
                      : ''
                  }`}
                  onClick={() =>
                    handleOptionSelect(option[keyProperty as keyof typeof option] as number)
                  }
                  onMouseEnter={() => setActiveIndex(index)}
                  role="option"
                  id={`${id}-option-${index}`}
                  aria-selected={index === activeIndex}
                >
                  {String(option[displayProperty as keyof typeof option])}
                </div>
              ))
            )}
          </div>
        )}
      </div>
      {error && <p className="text-red-600 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default SearchableDropdown;